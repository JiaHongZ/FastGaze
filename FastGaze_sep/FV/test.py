from typing import Optional, List
from timeit import default_timer as timer
import argparse
from datetime import datetime
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import pickle
from metrics import compute_NSS, compute_cAUC, compute_mm
from models import VS
from fastgaze import fastgaze
from utils import seed_everything, fixations2seq, get_args_parser_test, save_model_train
from dataset import fixation_dataset, COCOSearch18Collator
from scipy.ndimage import gaussian_filter

torch.autograd.set_detect_anomaly(True)
def generate_saliency_map_from_fixations(image_shape, fixation_points, gaussian_sigma=10, visualize=False):
    """
    Generate a saliency map from fixation points using Gaussian filtering.

    Parameters:
    - image_shape: tuple (H, W) of the target image.
    - fixation_points: list of tuples [(x1, y1), (x2, y2), ...].
    - gaussian_sigma: float, standard deviation for the Gaussian filter.
    - visualize: bool, whether to visualize the fixation points and saliency map.

    Returns:
    - saliency_map: 2D array representing the saliency map.
    """
    # Create an empty saliency map
    saliency_map = np.zeros(image_shape, dtype=np.float32)

    # Place fixation points as impulses
    for x, y in fixation_points:
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            saliency_map[y, x] += 1.0

    # Apply Gaussian filter
    saliency_map = gaussian_filter(saliency_map, sigma=gaussian_sigma)

    # Optionally visualize the results
    if visualize:
        plt.figure(figsize=(10, 5))

        # Original fixation points
        plt.subplot(1, 3, 1)
        plt.title("Fixation Points")
        plt.imshow(np.zeros(image_shape), cmap='gray')
        for x, y in fixation_points:
            plt.plot(x, y, 'ro')  # Plot points as red dots, x对应h, y对应w
        # Saliency map
        plt.subplot(1, 3, 2)
        plt.title("Saliency Map")
        plt.imshow(saliency_map, cmap='hot')
        
        plt.subplot(1, 3, 3)
        plt.colorbar()
        plt.savefig('Saliency.png')

        plt.tight_layout()
        plt.show()

    return saliency_map

def run_model(model, src, task, device = "cuda:0", im_h=20, im_w=32, patch_size = 16, num_samples = 1, batch_type=0):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h//2)*patch_size, (im_w//2)*patch_size]).unsqueeze(0).repeat(num_samples, 1)
    if batch_type == torch.Tensor([[2]]):
        mask = 0
    else:
        mask = 1
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src = src, tgt = firstfix, task = task * mask)
    token_prob = token_prob.detach().cpu().numpy()
    ys = ys.cpu().detach().numpy()
    xs = xs.cpu().detach().numpy()
    ts = ts.cpu().detach().numpy()
    scanpaths = []
    for i in range(num_samples):
        ys_i = [(im_h//2) * patch_size] + list(ys[:, i, 0])[1:]
        xs_i = [(im_w//2) * patch_size] + list(xs[:, i, 0])[1:]
        ts_i = list(ts[:, i, 0])
        token_type = [0] + list(np.argmax(token_prob[:, i, :], axis=-1))[1:]
        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(im_h * patch_size - 2, y),min(im_w * patch_size - 2, x), t])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths

def test_TP(args,model):
    args.condition = 'present'
    
    from metrics_tp import postprocessScanpaths, get_seq_score, get_seq_score_time, get_semantic_seq_score, get_semantic_seq_score_time, get_ed, get_ed_time, get_semantic_ed, get_semantic_ed_time
    device = torch.device('cuda:{}'.format(args.cuda))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TP_test.json'
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition']==args.condition, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    
    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join('/zjh/data/FastGaze-cocosearch', 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    for target_traj in test_task_img_pairs:
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir[0], task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, batch_type=torch.Tensor([[0]]))
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('/zjh/data/FastGaze-cocosearch', 'clusters_TP.npy'), allow_pickle=True).item()


    for traj in predictions:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs = torch.stack((s2x, s2y), dim=1)
        pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
        saliency_map = generate_saliency_map_from_fixations((320,512),pre_fixs,gaussian_sigma=25,visualize=False) 
        traj['saliency_map'] = saliency_map

    nss = compute_NSS(human_scanpaths, predictions)
    mm = compute_mm(human_scanpaths, predictions, 512, 320)
    
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    with open('/zjh/data/FastGaze-cocosearch/semantic_seq_full/test_TP_Sem.pkl', "rb") as r:
        fixations_dict = pickle.load(r)
        r.close()
    sem_seq_score = get_semantic_seq_score(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    sem_seq_score_t = get_semantic_seq_score_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    ed = get_ed(predictions, fix_clusters, max_len)
    ed_t = get_ed_time(predictions, fix_clusters, max_len, t_dict)
    sem_ed = get_semantic_ed(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    sem_ed_t = get_semantic_ed_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    # print(seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t)
    return seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t, nss, np.array(mm) ,np.array(mm).mean()
    
def test_TA(args,model):
    # type 0 TP, 1 TA, 2 FV
    
    args.condition = 'absent'
    
    from metrics_ta import postprocessScanpaths, get_seq_score, get_seq_score_time, get_semantic_seq_score, get_semantic_seq_score_time, get_ed, get_ed_time, get_semantic_ed, get_semantic_ed_time
    device = torch.device('cuda:{}'.format(args.cuda))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TA_test.json'
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition']==args.condition, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    
    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join('/zjh/data/FastGaze-cocosearch', 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    for target_traj in test_task_img_pairs:
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir[1], task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, batch_type=torch.Tensor([[1]]))
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('/zjh/data/FastGaze-cocosearch', 'clusters.npy'), allow_pickle=True).item()
    
    for traj in predictions:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs = torch.stack((s2x, s2y), dim=1)
        pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
        saliency_map = generate_saliency_map_from_fixations((320,512),pre_fixs,gaussian_sigma=25,visualize=False) 
        traj['saliency_map'] = saliency_map

    nss = compute_NSS(human_scanpaths, predictions)
    mm = compute_mm(human_scanpaths, predictions, 512, 320)
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    with open('/zjh/data/FastGaze-cocosearch/semantic_seq_full/test_TA_Sem.pkl', "rb") as r:
        fixations_dict = pickle.load(r)
        r.close()
    ed = get_ed(predictions, fix_clusters, max_len)
    ed_t = get_ed_time(predictions, fix_clusters, max_len, t_dict)
    
    if args.condition == 'freeview':
        sem_seq_score, sem_seq_score_t, sem_ed, sem_ed_t = 0, 0, 0, 0
    else:
        sem_seq_score = get_semantic_seq_score(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
        sem_seq_score_t = get_semantic_seq_score_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
        sem_ed = get_semantic_ed(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
        sem_ed_t = get_semantic_ed_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    return seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t, nss, np.array(mm) ,np.array(mm).mean()

def test_FV(args,model):
    args.condition = 'freeview'
    from metrics_fv import postprocessScanpaths, get_seq_score, get_semantic_seq_score, get_ed, get_semantic_ed
    from metrics import postprocessScanpaths
    device = torch.device('cuda:{}'.format(args.cuda))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_FV_test.json'
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition']==args.condition, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    
    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join('/zjh/data/FastGaze-cocosearch', 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    for target_traj in test_task_img_pairs:
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir[2], task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, batch_type=torch.Tensor([[2]]))
        for idx, scanpath in enumerate(scanpaths):
            # pred_list.append((name, condition, idx+1, scanpath))
            pred_list.append((task_name, name, condition, idx+1, scanpath))
    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('/zjh/data/FastGaze-cocosearch', 'clusters.npy'), allow_pickle=True).item()
    
    for traj in predictions:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs = torch.stack((s2x, s2y), dim=1)
        pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
        saliency_map = generate_saliency_map_from_fixations((320,512),pre_fixs,gaussian_sigma=25,visualize=False) 
        traj['saliency_map'] = saliency_map

    mm = compute_mm(human_scanpaths, predictions, 512, 320)
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    with open('/zjh/data/FastGaze-cocosearch/semantic_seq_full/test.pkl', "rb") as r:
        fixations_dict = pickle.load(r)
        r.close()
    
    ed = get_ed(predictions, fix_clusters, max_len)
    # ed_t = get_ed_time(predictions, fix_clusters, max_len, t_dict)
    
    return seq_score, ed, np.array(mm) ,np.array(mm).mean()
       
def train(epoch, args, model, SlowOpt, MidOpt, FastOpt, loss_fn_token, loss_task_token, loss_fn_y, loss_fn_x, loss_fn_t, train_dataloader, model_dir, model_name, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.train()
    token_losses = 0
    reg_losses = 0
    t_losses = 0
    task_losses = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix, batch_type in tepoch:
            # type 0 TP, 1 TA, 2 FV
            # 先将 mask 的形状扩展到 (b, 1, 1, ..., 1)，以便与 batch_tasks 广播， FV时，batch task需要为0
            mask = (batch_type != 2).float().view(-1, *[1] * (batch_tasks.dim() - 1))
            # print(batch_type, mask, batch_tasks * mask)
            
            out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks * mask)
            out_y, out_x = torch.clamp(out_y, min=0, max=im_h * patch_size - 2), torch.clamp(out_x, min=0, max=im_w * patch_size - 2)

            SlowOpt.zero_grad()
            MidOpt.zero_grad()
            FastOpt.zero_grad()

            tgt_out = batch_tgt.to(device)
            batch_tgt_padding_mask = batch_tgt_padding_mask.to(device)
            token_gt = batch_tgt_padding_mask.long()
            fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            #predict padding or valid fixation
            token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
            task_loss = loss_task_token(batch_type.to(device))
            
            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask
            #calculate regression L1 losses for only valid ground truth fixations
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            loss = task_loss + token_loss + reg_loss + t_loss
            loss.backward()
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()
            task_losses += task_loss.item()
            
            SlowOpt.step()
            MidOpt.step()
            FastOpt.step()
            
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch, task_loss=task_losses/minibatch)
            
    return token_losses / len(train_dataloader),  reg_losses / len(train_dataloader), t_losses / len(train_dataloader), task_losses / len(train_dataloader), SlowOpt, MidOpt, FastOpt, model_dir, model_name
    

def evaluate(model, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, valid_dataloader, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.eval()
    token_losses = 0
    reg_losses = 0
    t_losses = 0

    with tqdm(valid_dataloader, unit="batch") as tepoch:
        minibatch = 0

        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix, batch_type in tepoch:
            
            mask = (batch_type != 2).float().view(-1, *[1] * (batch_tasks.dim() - 1))
            with torch.no_grad():
                out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks * mask)
            out_y, out_x = torch.clamp(out_y, min=0, max=im_h *patch_size -2), torch.clamp(out_x, min=0, max=im_w *patch_size -2)

            tgt_out = batch_tgt.to(device)
            batch_tgt_padding_mask = batch_tgt_padding_mask.to(device)
            token_gt = batch_tgt_padding_mask.long()
            fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch)
            
    return token_losses / len(valid_dataloader),  reg_losses / len(valid_dataloader), t_losses/len(valid_dataloader)
    
    
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.cuda))
    args.img_ftrs_dir = ['/zjh/data/FastGaze-cocosearch/image_features_TP','/zjh/data/FastGaze-cocosearch/image_features','/zjh/data/FastGaze-cocosearch/image_features']
    
    # valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=4, collate_fn = collate_fn)

    vs = VS(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim,
    num_decoder_layers=args.num_decoder, dim_feedforward = args.hidden_dim, 
    img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device, img_ior=args.img_ior, img_mask=args.img_mask, sc_mask=args.sc_mask, txt_mask=args.txt_mask, sc_ior=args.sc_ior,  txt_ior=args.txt_ior, max_len = args.max_len).to(device)

    model = fastgaze(vs, spatial_dim = (args.im_h, args.im_w), max_len = args.max_len, device = device).to(device)
    
    trained_model = args.trained_model
    model.load_state_dict(torch.load(trained_model, map_location=device))
    model.eval()
    
    args.zerogaze = False
    args.num_samples = 10
    epoch = 1
    
    seq_score, fed, mm ,mmmean = test_FV(args,model)
    output_str2 = f"Epoch: {epoch}, FV: seq_score: {seq_score:.3f}, fed: {fed:.3f}, mm: {mm}, mmmean: {mmmean:.3f} \n"
    print(output_str2)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('FastGaze Train', parents=[get_args_parser_test()])
    args = parser.parse_args()
    main(args)
    

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
from metrics import *
import pickle

from models import VS
from fastgaze import fastgaze
from utils import seed_everything, fixations2seq, get_args_parser_train, save_model_train
from dataset import fixation_dataset, COCOSearch18Collator

torch.autograd.set_detect_anomaly(True)

def run_model(model, src, task, device = "cuda:0", im_h=20, im_w=32, patch_size = 16, num_samples = 1):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h//2)*patch_size, (im_w//2)*patch_size]).unsqueeze(0).repeat(num_samples, 1)
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src = src, tgt = firstfix, task = task)
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
    
    
def test(args,model):
    device = torch.device('cuda:{}'.format(args.cuda))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
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
    embedding_dict = np.load(open(join(dataset_root, 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    for target_traj in test_task_img_pairs:
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir, task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('/zjh/data/FastGaze-cocosearch', 'clusters_TP.npy'), allow_pickle=True).item()
    
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
    return seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t
    
    
def train(epoch, args, model, SlowOpt, MidOpt, FastOpt, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, train_dataloader, model_dir, model_name, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.train()
    token_losses = 0
    reg_losses = 0
    t_losses = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix in tepoch:
            out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks)
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
            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask
            #calculate regression L1 losses for only valid ground truth fixations
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            loss = token_loss + reg_loss + t_loss
            loss.backward()
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()

            SlowOpt.step()
            MidOpt.step()
            FastOpt.step()
            
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch)
    return token_losses / len(train_dataloader),  reg_losses / len(train_dataloader), t_losses / len(train_dataloader), SlowOpt, MidOpt, FastOpt, model_dir, model_name
    

def evaluate(model, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, valid_dataloader, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.eval()
    token_losses = 0
    reg_losses = 0
    t_losses = 0

    with tqdm(valid_dataloader, unit="batch") as tepoch:
        minibatch = 0

        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix in tepoch:
            with torch.no_grad():
                out_token, out_y, out_x,out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks)
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
    device_id = args.cuda
    retraining = args.retraining
    last_checkpoint = args.last_checkpoint
    args.model_root = ''
    if retraining:
        model_dir = '/'.join(args.last_checkpoint.split('/')[:-1])
        args = argparse.Namespace(**json.load(open(join(model_dir, 'config.json'))))
        logfile = 'logs/output_' + last_checkpoint.split('/')[-2].split('_')[-1]+'.txt'
        args.cuda = device_id
    else:
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") 
        model_dir = join('train_' + str(args.net_name)+'_'+str(args.num_encoder)+'E_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'_'+str(args.hidden_dim)+'_'+str(args.sc_ior)+'_'+str(args.sc_mask)+'_' + timenow)
        logfile = model_dir + '/log.txt'
        os.mkdir(model_dir)
        
        open(logfile, 'w').close()
        with open(logfile, "a") as myfile:
            myfile.write(str(vars(args)) + '\n\n')
            myfile.close()
    print(str(vars(args)) + '\n\n')
    with open(join(model_dir, 'config.json'), "w") as outfile:
        json.dump(vars(args), outfile)
        outfile.close()


    model_name = str(args.net_name)+'_'+str(args.num_encoder)+'E_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'_'+str(args.hidden_dim)+'d'
    dataset_root = args.dataset_dir
    train_file = args.train_file
    valid_file = args.valid_file
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)

        
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
            
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)

    train_dataset = fixation_dataset(seq_train, img_ftrs_dir = args.img_ftrs_dir)
    valid_dataset = fixation_dataset(seq_valid, img_ftrs_dir = args.img_ftrs_dir)

    #target embeddings
    embedding_dict = np.load(open(join(dataset_root, 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()

    collate_fn = COCOSearch18Collator(embedding_dict, args.max_len, args.im_h, args.im_w, args.patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=6, collate_fn = collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=6, collate_fn = collate_fn)

    vs = VS(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim,
    num_decoder_layers=args.num_decoder, encoder_dropout = args.encoder_dropout, decoder_dropout = args.decoder_dropout, dim_feedforward = args.hidden_dim, 
    img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device, img_ior=args.img_ior, img_mask=args.img_mask, sc_mask=args.sc_mask, txt_mask=args.txt_mask, sc_ior=args.sc_ior,  txt_ior=args.txt_ior, max_len = args.max_len).to(device)

    model = fastgaze(vs, spatial_dim = (args.im_h, args.im_w), dropout=args.cls_dropout, max_len = args.max_len, device = device).to(device)

    loss_fn_token = torch.nn.NLLLoss()
    loss_fn_y = nn.L1Loss(reduction='none')
    loss_fn_x = nn.L1Loss(reduction='none')
    loss_fn_t = nn.L1Loss(reduction='none')
    #Disjoint optimization
    head_params = list(model.vs.lip.parameters()) + list(model.token_predictor.parameters())
    SlowOpt = torch.optim.AdamW(head_params, lr=args.head_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    belly_params = list(model.fef.generator_t_mu.parameters()) + list(model.fef.generator_t_logvar.parameters())
    MidOpt = torch.optim.AdamW(belly_params, lr=args.belly_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    tail_params = list(model.vs.sc.parameters()) + list(model.fef.generator_y_mu.parameters()) + list(model.fef.generator_x_mu.parameters()) + list(model.fef.generator_y_logvar.parameters()) + list(model.fef.generator_x_logvar.parameters()) + list(model.querypos_embed.parameters())
    FastOpt = torch.optim.AdamW(tail_params, lr=args.tail_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    start_epoch = 1
    if retraining:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        SlowOpt.load_state_dict(checkpoint['optim_slow'])
        MidOpt.load_state_dict(checkpoint['optim_mid'])
        FastOpt.load_state_dict(checkpoint['optim_fast'])
        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)
    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        train_token_loss, train_reg_loss, train_t_loss, SlowOpt, MidOpt, FastOpt, model_dir, model_name = train(epoch = epoch, args = args, model = model, SlowOpt = SlowOpt, FastOpt = FastOpt, MidOpt = MidOpt, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t = loss_fn_t, train_dataloader = train_dataloader, model_dir = model_dir, model_name = model_name, device = device)
        end_time = timer()
        
        valid_token_loss, valid_reg_loss, valid_t_loss = evaluate(model = model, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t=loss_fn_t, valid_dataloader = valid_dataloader, device = device)
        
        args.condition = 'present'
        args.zerogaze = False
        args.num_samples = 10
        seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t = test(args,model)
        
        output_str = f"Epoch: {epoch}, Train token loss: {train_token_loss:.3f}, Train reg loss: {train_reg_loss:.3f}, Train T loss: {train_t_loss:.3f}, Val token loss: {valid_token_loss:.3f},  Val reg loss: {valid_reg_loss:.3f}, Valid T loss: {valid_t_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"
        output_str2 = f"Epoch: {epoch}, seq_score: {seq_score:.3f}, seq_score_t: {seq_score_t:.3f}, sem_seq_score: {sem_seq_score:.3f}, sem_seq_score_t: {sem_seq_score_t:.3f}, ed: {ed:.3f}, ed_t: {ed_t:.3f}, sem_ed: {sem_ed:.3f}, sem_ed_t: {sem_ed_t:.3f}\n"
        if seq_score >= 0.497:
            save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name)
        # print(output_str)
        with open(logfile, "a") as myfile:
            myfile.write(output_str)
            myfile.write(output_str2)
            myfile.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('FastGaze Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)
    

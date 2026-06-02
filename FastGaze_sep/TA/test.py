import argparse
from os.path import join
import json
import numpy as np
import torch

from models import VS
from fastgaze import fastgaze
from utils import seed_everything, get_args_parser_test
from metrics import *
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings("ignore")



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
    
    
def test(args):
    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    vs = VS(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim, num_decoder_layers=args.num_decoder, dim_feedforward = args.hidden_dim, img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device, img_ior=args.img_ior, img_mask=args.img_mask, sc_mask=args.sc_mask, txt_mask=args.txt_mask, sc_ior=args.sc_ior,  txt_ior=args.txt_ior, max_len = args.max_len).to(device)
    model = fastgaze(vs, spatial_dim = (args.im_h, args.im_w), max_len = args.max_len, device = device).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
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
    fix_clusters = np.load(join('./data', 'clusters.npy'), allow_pickle=True).item()
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    if args.condition == 'present':
        with open('/zjh/data/FastGaze-cocosearch/SemSS/test_TP_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    elif args.condition == 'absent':
        with open('/zjh/data/FastGaze-cocosearch/SemSS/test_TA_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    sem_seq_score = get_semantic_seq_score(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/SemSS/stuffthing_maps')
    sem_seq_score_t = get_semantic_seq_score_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/SemSS/stuffthing_maps')
    ed = get_ed(predictions, fix_clusters, max_len)
    ed_t = get_ed_time(predictions, fix_clusters, max_len, t_dict)
    sem_ed = get_semantic_ed(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/SemSS/stuffthing_maps')
    sem_ed_t = get_semantic_ed_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/SemSS/stuffthing_maps')
    # print(seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t)
    return seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t
    
def main(args):
    seed_everything(args.seed)
    seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t = test(args)
    print('Sequence Score : {:.3f}, Sequence Score with Duration : {:.3f}'.format(seq_score, seq_score_t))
    # print('Sem Sequence Score : {:.3f}, Sem Sequence Score with Duration : {:.3f}'.format(sem_seq_score, sem_seq_score_t))
    print(seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('FastGaze Test', parents=[get_args_parser_test()])
    args = parser.parse_args()
    main(args)

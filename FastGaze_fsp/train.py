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
from metrics_tp import *
from models import VS
from fastgaze import fastgaze
from utils import seed_everything, fixations2seq, get_args_parser_train, save_model_train
from dataset import fixation_dataset, COCOSearch18Collator

torch.autograd.set_detect_anomaly(True)

def run_model(model, src, task, device = "cuda:0", im_h=20, im_w=32, patch_size = 16, num_samples = 1, batch_type=0):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h//2)*patch_size, (im_w//2)*patch_size]).unsqueeze(0).repeat(num_samples, 1)
    if batch_type == torch.Tensor([[2]]):
        task = task * 0
    with torch.no_grad():
        out_task, token_prob, ys, xs, ts = model(src = src, tgt = firstfix, task = task, type_tensor=batch_type)
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
def test_TP_simple(args,model):
    # type 0 TP, 1 TA, 2 FV
    
    args.condition = 'present'
    
    from metrics_tp import get_seq_score
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
    
    print('最终测试用10,Generating {} scanpaths per test case...'.format(3))
    for target_traj in test_task_img_pairs:
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir[0], task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=3, batch_type=torch.Tensor([[0]]))
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('/zjh/data/FastGaze-cocosearch', 'clusters_TP.npy'), allow_pickle=True).item()
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    return seq_score
     
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
    # print(seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t)
    return seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t

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
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    
    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join('/zjh/data/FastGaze-cocosearch', 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
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
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    # print("Calculating Sequence Score with Duration...")
    # seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    with open('/zjh/data/FastGaze-cocosearch/semantic_seq_full/test.pkl', "rb") as r:
        fixations_dict = pickle.load(r)
        r.close()
    
    ed = get_ed(predictions, fix_clusters, max_len)
    # ed_t = get_ed_time(predictions, fix_clusters, max_len, t_dict)
    
    sem_seq_score = get_semantic_seq_score(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    # sem_seq_score_t = get_semantic_seq_score_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    sem_ed = get_semantic_ed(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    # sem_ed_t = get_semantic_ed_time(predictions, fixations_dict, max_len, '/zjh/data/FastGaze-cocosearch/semantic_seq_full/stuffthing_maps')
    # print(seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t)
    return seq_score, sem_seq_score, ed, sem_ed


    
# 高斯核生成
def gaussian_kernel(size, sigma):
    kernel = torch.Tensor([[torch.exp(-torch.tensor((x - size//2)**2 + (y - size//2)**2) / (2 * sigma ** 2))
                            for x in range(size)] for y in range(size)])
    kernel = kernel / kernel.sum()  # Normalize to ensure sum is 1
    return kernel
# 高斯平滑
def apply_gaussian_smoothing(saliency_map, kernel_size=15, sigma=3.0):
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to(saliency_map.device)  # Shape (1, 1, kernel_size, kernel_size)
    # 扩展通道维度用于卷积
    saliency_map = saliency_map.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
    # 进行卷积
    smoothed_saliency_map = F.conv2d(saliency_map, kernel, padding=kernel_size//2)
    # 恢复回去
    smoothed_saliency_map = smoothed_saliency_map.squeeze(1)  # Shape: (batch_size, height, width)
    return smoothed_saliency_map

def generate_saliency_map_with_smoothing(pred_x, pred_y, tgt_x, tgt_y, img_size, kernel_size=15, sigma=3.0):
    saliency_map_pred = torch.zeros((pred_x.shape[0], img_size[0], img_size[1]), device=pred_x.device)
    saliency_map_tgt = torch.zeros_like(saliency_map_pred)
    
    # For each predicted and target fixation point, place a Gaussian blob
    for i in range(pred_x.shape[0]):
        # Ensure that indices are integers (long type)
        pred_x_idx = pred_x[i].long()
        pred_y_idx = pred_y[i].long()
        tgt_x_idx = tgt_x[i].long()
        tgt_y_idx = tgt_y[i].long()

        # Set the center of the Gaussian at the predicted and target locations
        saliency_map_pred[i, pred_y_idx, pred_x_idx] = 1.0
        saliency_map_tgt[i, tgt_y_idx, tgt_x_idx] = 1.0
    
    # Apply Gaussian smoothing to each saliency map to create a Gaussian blob for each fixation
    saliency_map_pred = apply_gaussian_smoothing(saliency_map_pred, kernel_size, sigma)
    saliency_map_tgt = apply_gaussian_smoothing(saliency_map_tgt, kernel_size, sigma)
    
    return saliency_map_pred, saliency_map_tgt
# 注视损失计算（focal loss）
def fixation_loss(pred_map, tgt_map, alpha=2, beta=4):
    # Clip pred_map to avoid log(0) issues
    epsilon = 1e-6
    pred_map = torch.clamp(pred_map, min=epsilon, max=1 - epsilon)
    # 使用焦点损失公式计算
    focal_loss = -(tgt_map * (1 - pred_map) ** alpha * torch.log(pred_map) + 
                   (1 - tgt_map) * (pred_map ** beta) * torch.log(1 - pred_map))
    return focal_loss.mean()
def save_saliency_maps(saliency_pred, saliency_tgt, filename_pred="saliency_pred.png", filename_tgt="saliency_tgt.png"):
    # Create the figure
    saliency_pred = saliency_pred[0]  # Take the first image in the batch
    saliency_tgt = saliency_tgt[0]  # Take the first target image in the batch
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Predicted saliency map
    axes[0].imshow(saliency_pred.cpu().detach().numpy(), cmap='hot')
    axes[0].set_title('Predicted Saliency Map')
    axes[0].axis('off')

    # Target saliency map
    axes[1].imshow(saliency_tgt.cpu().detach().numpy(), cmap='hot')
    axes[1].set_title('Target Saliency Map')
    axes[1].axis('off')

    # Save the image
    plt.savefig(filename_pred, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(filename_tgt, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory

# 示例: 生成并保存显著图
def generate_and_save_saliency(pred_x, pred_y, tgt_x, tgt_y, img_size, kernel_size=15, sigma=3.0, save_dir="saliency_maps/"):
    saliency_pred, saliency_tgt = generate_saliency_map_with_smoothing(pred_x, pred_y, tgt_x, tgt_y, img_size, kernel_size, sigma)
    save_saliency_maps(saliency_pred, saliency_tgt, filename_pred=save_dir + "saliency_pred.png", filename_tgt=save_dir + "saliency_tgt.png")

# 定义分散损失函数：鼓励预测的眼动点相互远离
def dispersion_loss(pred_x, pred_y, mask=None, epsilon=1e-6, y_weight=1.0):
    """
    pred_x, pred_y: shape (batch_size, N) representing predicted gaze point x and y coordinates (in pixels).
    mask: Optional, shape (batch_size, N), with 1 indicating valid prediction points and 0 indicating invalid points (e.g., padding).
    epsilon: Small constant to prevent division by zero.
    y_weight: Weight applied to y distances to ensure vertical separation is also encouraged.
    
    The returned loss value is larger when points are closer together, and zero when points are sufficiently far apart.
    """
    batch_size, N = pred_x.shape
    total_loss = 0.0
    valid_samples = 0

    for b in range(batch_size):
        # If mask is provided, only consider valid prediction points
        if mask is not None:
            valid_idx = (mask[b] > 0).nonzero(as_tuple=True)[0]
            xs = pred_x[b][valid_idx]
            ys = pred_y[b][valid_idx]
        else:
            xs = pred_x[b]
            ys = pred_y[b]
        
        # Skip samples with less than two valid points
        if xs.numel() < 2:
            continue
        
        # Stack the x and y coordinates into a (n_valid, 2) tensor
        coords = torch.stack([xs, ys], dim=1)  # Shape: (n_valid, 2)
        n_valid = coords.shape[0]
        
        # Compute pairwise Euclidean distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff**2, dim=2) + epsilon)
        
        # Apply weight to y-distance (vertical separation encouragement)
        distances = distances + (y_weight - 1.0) * torch.abs(diff[:, :, 1])
        
        # Select the upper triangular part of the distance matrix (excluding diagonal)
        i, j = torch.triu_indices(n_valid, n_valid, offset=1)
        dists = distances[i, j]
        
        # Compute loss based on exponential decay of distances
        sample_loss = torch.exp(-dists).sum()
        total_loss += sample_loss
        valid_samples += 1
    
    # Average the loss over valid samples
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=pred_x.device)

def proximity_loss(pred_x, pred_y, mask=None, threshold=10.0, epsilon=1e-6):
    """
    pred_x, pred_y: shape (batch_size, N) 表示预测的眼动点的 x 和 y 坐标（单位：像素）。
    mask: 可选，形状 (batch_size, N)，为 1 表示有效的预测点，为 0 表示无效（例如 padding）。
    threshold: 眼动点之间的最小距离阈值，单位为像素，低于该值则会计算损失。
    epsilon: 防止除零的小常数。
    
    返回的损失值越大，说明预测点之间距离越近，损失值为 0 时表示所有点彼此都足够远。
    """
    batch_size, N = pred_x.shape
    total_loss = 0.0
    valid_samples = 0
    for b in range(batch_size):
        # 如果提供 mask，则只考虑有效的预测点
        if mask is not None:
            valid_idx = (mask[b] > 0).nonzero(as_tuple=True)[0]
            xs = pred_x[b][valid_idx]
            ys = pred_y[b][valid_idx]
        else:
            xs = pred_x[b]
            ys = pred_y[b]
        
        # 如果有效点不足两个，则跳过该样本
        if xs.numel() < 2:
            continue
        
        # 组合成 (n_valid, 2) 的坐标张量
        coords = torch.stack([xs, ys], dim=1)  # (n_valid, 2)
        n_valid = coords.shape[0]
        
        # 计算两两之间的欧氏距离
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        distances = torch.sqrt(torch.sum(diff**2, dim=2) + epsilon)
        
        # 取上三角部分（不包括对角线，即不重复计算和排除自身距离为0的情况）
        i, j = torch.triu_indices(n_valid, n_valid, offset=1)
        dists = distances[i, j]
        
        # 计算距离小于阈值的损失项：距离越小，损失越大
        close_dists = dists[dists < threshold]
        
        # 使用 exp(-distance) 作为损失项
        sample_loss = torch.exp(-close_dists).sum() if close_dists.numel() > 0 else 0
        total_loss += sample_loss
        valid_samples += 1
    
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=pred_x.device)
    
      
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
            
            out_task, out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks * mask, type_tensor=batch_type)
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
            task_loss = loss_task_token(out_task, batch_type.to(device))
            
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
                out_task, out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks * mask, type_tensor=batch_type)
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
        # model_dir = join('train_' + 'fastgaze_'+str(args.num_encoder)+'E_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'_'+str(args.hidden_dim)+'_' + timenow)
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
    args.num_samples = 10
    
    seq_trains = []
    seq_valids = []
    train_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TP_train.json'
    valid_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TP_validation.json'
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)
    seq_trains.append(seq_train)
    seq_valids.append(seq_valid)
    train_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TA_train.json'
    valid_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_TA_valid.json'
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)
    seq_trains.append(seq_train)
    seq_valids.append(seq_valid)
    train_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_FV_train.json'
    valid_file = '/zjh/data/FastGaze-cocosearch/coco_search18_fixations_FV_valid.json'
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)
    seq_trains.append(seq_train)
    seq_valids.append(seq_valid)
    
    args.img_ftrs_dir = ['/zjh/data/FastGaze-cocosearch/image_features_TP','/zjh/data/FastGaze-cocosearch/image_features','/zjh/data/FastGaze-cocosearch/image_features']
    train_dataset = fixation_dataset(seq_trains, img_ftrs_dir = args.img_ftrs_dir)
    valid_dataset = fixation_dataset(seq_valids, img_ftrs_dir = args.img_ftrs_dir)
    #target embeddings
    embedding_dict = np.load(open(join('/zjh/data/FastGaze-cocosearch', 'clip_embeddings.npy'), mode='rb'), allow_pickle = True).item()

    collate_fn = COCOSearch18Collator(embedding_dict, args.max_len, args.im_h, args.im_w, args.patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=4, collate_fn = collate_fn, sampler=train_dataset.get_sampler())
    # valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=4, collate_fn = collate_fn)

    vs = VS(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim,
    num_decoder_layers=args.num_decoder, encoder_dropout = args.encoder_dropout, decoder_dropout = args.decoder_dropout, dim_feedforward = args.hidden_dim, 
    img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device, img_ior=args.img_ior, img_mask=args.img_mask, sc_mask=args.sc_mask, txt_mask=args.txt_mask, sc_ior=args.sc_ior,  txt_ior=args.txt_ior, max_len = args.max_len).to(device)

    model = fastgaze(vs, spatial_dim = (args.im_h, args.im_w), dropout=args.cls_dropout, max_len = args.max_len, device = device).to(device)

    loss_task_token = torch.nn.CrossEntropyLoss()
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
    best_epo = 1
    best_seq = 0
    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        
        if epoch == 190: # 最后10轮加载之前最好的
            last_checkpoint = join(model_dir, model_name+'_'+str(best_epo)+'.pkg')
            checkpoint = torch.load(last_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
        
        train_token_loss, train_reg_loss, train_t_loss, task_loss, SlowOpt, MidOpt, FastOpt, model_dir, model_name = train(epoch = epoch, args = args, model = model, SlowOpt = SlowOpt, FastOpt = FastOpt, MidOpt = MidOpt, loss_fn_token = loss_fn_token, loss_task_token = loss_task_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t = loss_fn_t, train_dataloader = train_dataloader, model_dir = model_dir, model_name = model_name, device = device)
        end_time = timer()
        
        # valid_token_loss, valid_reg_loss, valid_t_loss = evaluate(model = model, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t=loss_fn_t, valid_dataloader = valid_dataloader, device = device)
        
        args.zerogaze = False
        args.num_samples = 10
        # output_str = f"Epoch: {epoch}, Train task loss: {task_loss:.3f}, Train token loss: {train_token_loss:.3f}, Train reg loss: {train_reg_loss:.3f}, Train T loss: {train_t_loss:.3f}, Val token loss: {valid_token_loss:.3f},  Val reg loss: {valid_reg_loss:.3f}, Valid T loss: {valid_t_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"        
        output_str = f"Epoch: {epoch}, Train task loss: {task_loss:.3f}, Train token loss: {train_token_loss:.3f}, Train reg loss: {train_reg_loss:.3f}, Train T loss: {train_t_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"        
        with open(logfile, "a") as myfile:
            myfile.write(output_str)
        
        seq_score = test_TP_simple(args,model)
        
        output_str2 = f"Epoch: {epoch}, best_epo: {best_epo}, bestss: {best_seq:.3f}, TP: seq_score: {seq_score:.3f},\n"
        with open(logfile, "a") as myfile:
            myfile.write(output_str2)
            myfile.close()
            
        if seq_score >= best_seq:
            best_seq = seq_score
            best_epo = epoch            
        if seq_score >= 0.490:
            seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t = test_TP(args,model)
            output_str2 = f"Epoch: {epoch}, TP: seq_score: {seq_score:.3f}, seq_score_t: {seq_score_t:.3f}, sem_seq_score: {sem_seq_score:.3f}, sem_seq_score_t: {sem_seq_score_t:.3f}, ed: {ed:.3f}, ed_t: {ed_t:.3f}, sem_ed: {sem_ed:.3f}, sem_ed_t: {sem_ed_t:.3f}\n"
            with open(logfile, "a") as myfile:
                myfile.write(output_str2)
                myfile.close()
            save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name)
        
            seq_score,seq_score_t, sem_seq_score, sem_seq_score_t, ed, ed_t, sem_ed, sem_ed_t = test_TA(args,model)
            output_str2 = f"Epoch: {epoch}, TA: seq_score: {seq_score:.3f}, seq_score_t: {seq_score_t:.3f}, sem_seq_score: {sem_seq_score:.3f}, sem_seq_score_t: {sem_seq_score_t:.3f}, ed: {ed:.3f}, ed_t: {ed_t:.3f}, sem_ed: {sem_ed:.3f}, sem_ed_t: {sem_ed_t:.3f}\n"
            with open(logfile, "a") as myfile:
                myfile.write(output_str2)
                
            seq_score, sem_seq_score, ed, sem_ed = test_FV(args,model)
            output_str2 = f"Epoch: {epoch}, FV: seq_score: {seq_score:.3f}, sem_seq_score: {sem_seq_score:.3f}, ed: {ed:.3f}, sem_ed: {sem_ed:.3f}\n"
        
        with open(logfile, "a") as myfile:
            myfile.write(output_str2)
            myfile.close()
            

        # print(output_str)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('FastGaze Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)
    

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import json
from os.path import isfile
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
import cv2 as cv
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
seed_everything(42)

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
        # plt.savefig('aa/scanpath'+'_'+args.condition+'_'+args.trained_model+'_'+args.task+'_'+args.imgfile+'saliency.png')

        plt.tight_layout()
        plt.show()

    return saliency_map

def run_model(model, src, task, device="cuda:4", im_h=20, im_w=32, patch_size=16, num_samples=1, type_tensor=torch.Tensor([[0]]), scanpaths=None):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h // 2) * patch_size, (im_w // 2) * patch_size]).unsqueeze(0).repeat(num_samples, 1)

    if scanpaths != None and args.condition == 'freeview':
        firstfix = torch.tensor([int(scanpaths[0][-1][0]), int(scanpaths[0][-1][1])]).unsqueeze(0).repeat(num_samples, 1)
    if type_tensor == torch.Tensor([[2]]):
        task = task * 0
    with torch.no_grad():
        _, token_prob, ys, xs, ts = model(src=src, tgt=firstfix, task=task, type_tensor=type_tensor)
    token_prob = token_prob.detach().cpu().numpy()
    ys = ys.cpu().detach().numpy()
    xs = xs.cpu().detach().numpy()
    ts = ts.cpu().detach().numpy()
    scanpaths = []
    for i in range(num_samples):
        ys_i = [(im_h // 2) * patch_size] + list(ys[:, i, 0])[1:]
        xs_i = [(im_w // 2) * patch_size] + list(xs[:, i, 0])[1:]
        ts_i = list(ts[:, i, 0])
        token_type = [0] + list(np.argmax(token_prob[:, i, :], axis=-1))[1:]
        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(im_h * patch_size - 2, y), min(im_w * patch_size - 2, x), t])
            else:
                break
        scanpaths.append(np.array(scanpath))
        
    return scanpaths

def gethuman(args):

    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    vs = VS(num_encoder_layers=args.num_encoder, nhead=args.nhead, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, sc_mask=args.sc_mask, sc_ior=args.sc_ior).to(device)
    model = fastgaze(transformer=vs, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len,
                       device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    typetensor = torch.Tensor([[0]])
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
        typetensor = torch.Tensor([[1]])
        print('use absent')
    if args.condition == 'freeview':
        fixation_path = join(dataset_root, 'coco_search18_fixations_FV_test.json')
        typetensor = torch.Tensor([[2]])
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(
        filter(lambda x: x['condition'] == args.condition and x['name'] == args.imgfile and x['task'] == args.task, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    test_task_img_pairs = np.unique(
        [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    
    # print(test_target_trajs)
    # for traj in test_target_trajs:
    #     s2x = np.array(traj['X'])
    #     s2y = np.array(traj['Y'])
    #     s2x = torch.from_numpy(s2x)
    #     s2y = torch.from_numpy(s2y)
    #     pre_fixs = torch.stack((s2x, s2y), dim=1)
    #     pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
    return test_target_trajs


def test(args,image_ftrs):

    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    vs = VS(num_encoder_layers=args.num_encoder, nhead=args.nhead, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, sc_mask=args.sc_mask, sc_ior=args.sc_ior).to(device)
    model = fastgaze(transformer=vs, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len,
                       device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    typetensor = torch.Tensor([[0]])
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
        typetensor = torch.Tensor([[1]])
        print('use absent')
    if args.condition == 'freeview':
        fixation_path = join(dataset_root, 'coco_search18_fixations_FV_test.json')
        typetensor = torch.Tensor([[2]])
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(
        filter(lambda x: x['split'] == 'test' and x['condition'] == args.condition and x['name'] == args.imgfile and x['task'] == args.task, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    test_task_img_pairs = np.unique(
        [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    
    print(test_target_trajs)
    embedding_dict = np.load(open(join(dataset_root, 'clip_embeddings.npy'), mode='rb'), allow_pickle=True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    task_name = args.task
    # for target_traj in test_task_img_pairs:
    task_emb = embedding_dict[task_name]
    
    scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, type_tensor=typetensor)
    
    for idx, scanpath in enumerate(scanpaths):
        pred_list.append((task_name, args.imgfile, args.condition, idx + 1, scanpath))
    predictions = postprocessScanpaths(pred_list)

    for traj in predictions:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs = torch.stack((s2x, s2y), dim=1)
        pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
        # saliency_map = generate_saliency_map_from_fixations((320,512),pre_fixs,gaussian_sigma=25,visualize=True) 
    return predictions


def test_fv(args):

    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    vs = VS(num_encoder_layers=args.num_encoder, nhead=args.nhead, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, sc_mask=args.sc_mask, sc_ior=args.sc_ior).to(device)
    model = fastgaze(transformer=vs, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len,
                       device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    typetensor = torch.Tensor([[0]])
    if args.condition == 'absent':
        print('not support use absent')
    if args.condition == 'freeview':
        fixation_path = join(dataset_root, 'coco_search18_fixations_FV_test.json')
        typetensor = torch.Tensor([[2]])
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(
        filter(lambda x: x['split'] == 'test' and x['condition'] == args.condition and x['task'] == args.task, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])

    test_task_img_pairs = np.unique(
        [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join(dataset_root, 'clip_embeddings.npy'), mode='rb'), allow_pickle=True).item()

    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    task_name = args.task
    # for target_traj in test_task_img_pairs:
    task_emb = embedding_dict[task_name]
    image_ftrs = torch.load('/zjh/data/FastGaze-cocosearch/image_features/'+args.task+'/'+args.imgfile[:-4]+'.pth').unsqueeze(0)
    
    scanpaths1 = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, type_tensor=typetensor)

    if args.condition == 'freeview':
        scanpaths2 = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, type_tensor=typetensor, scanpaths=scanpaths1)
    pred_list1 = []
    pred_list2 = []
    for idx, scanpath in enumerate(scanpaths1):
        pred_list1.append((task_name, args.imgfile, args.condition, idx + 1, scanpath))
    for idx, scanpath in enumerate(scanpaths2):
        pred_list2.append((task_name, args.imgfile, args.condition, idx + 1, scanpath))
    predictions1 = postprocessScanpaths(pred_list1)
    predictions2 = postprocessScanpaths(pred_list2)    
    print(len(predictions1))
    
    for traj in predictions1:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs = torch.stack((s2x, s2y), dim=1)
        pre_fixs = [(int(x), int(y)) for x, y in pre_fixs.tolist()]
    for traj in predictions2:
        s2x = np.array(traj['X'])
        s2y = np.array(traj['Y'])
        s2x = torch.from_numpy(s2x)
        s2y = torch.from_numpy(s2y)
        pre_fixs2 = torch.stack((s2x, s2y), dim=1)
        pre_fixs2 = [(int(x), int(y)) for x, y in pre_fixs2.tolist()]
        pre_fixs = pre_fixs.extend(pre_fixs2)
        # saliency_map = generate_saliency_map_from_fixations((320,512),pre_fixs,gaussian_sigma=25,visualize=True) 
    
    
    return predictions1, predictions2

# def test(args):

#     trained_model = args.trained_model
#     device = torch.device('cuda:{}'.format(args.cuda))
#     vs = VS(num_encoder_layers=args.num_encoder, nhead=args.nhead, d_model=args.hidden_dim,
#                               num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
#                               img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, sc_mask=args.sc_mask, sc_ior=args.sc_ior).to(device)
#     model = fastgaze(transformer=vs, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len,
#                        device=device).to(device)
#     print(model)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     dataset_root = args.dataset_dir
#     img_ftrs_dir = args.img_ftrs_dir
#     max_len = args.max_len
#     fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
#     typetensor = torch.Tensor([[0]])
#     if args.condition == 'absent':
#         fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
#         typetensor = torch.Tensor([[1]])
#         print('use absent')
#     if args.condition == 'freeview':
#         fixation_path = join(dataset_root, 'coco_search18_fixations_FV_test.json')
#         typetensor = torch.Tensor([[2]])
#     with open(fixation_path) as json_file:
#         human_scanpaths = json.load(json_file)
#     test_target_trajs = list(
#         filter(lambda x: x['split'] == 'test' and x['condition'] == args.condition and x['task'] == args.task, human_scanpaths))
#     if args.zerogaze:
#         test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
#         print("Zero Gaze on", args.task.replace('_', ' '))
#     t_dict = {}
#     for traj in test_target_trajs:
#         key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
#                                         traj['name'][:-4], traj['subject'])

#         t_dict[key] = np.array(traj['T'])

#     test_task_img_pairs = np.unique(
#         [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
#     embedding_dict = np.load(open(join(dataset_root, 'clip_embeddings.npy'), mode='rb'), allow_pickle=True).item()
#     pred_list = []
#     print('Generating {} scanpaths per test case...'.format(args.num_samples))
#     task_name = args.task
#     # for target_traj in test_task_img_pairs:
#     task_emb = embedding_dict[task_name]
#     image_ftrs = torch.load('/zjh/data/FastGaze-cocosearch/image_features/'+args.task+'/'+args.imgfile[:-4]+'.pth').unsqueeze(0)
#     scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples, type_tensor=typetensor)
#     for idx, scanpath in enumerate(scanpaths):
#         pred_list.append((task_name, args.imgfile, args.condition, idx + 1, scanpath))
#     predictions = postprocessScanpaths(pred_list)
#     return predictions

def convert_coordinate(X, Y, im_w, im_h):
    """
    convert from display coordinate to pixel coordinate

    X - x coordinate of the fixations
    Y - y coordinate of the fixations
    im_w - image width
    im_h - image height
    """
    display_w, display_h = 1680, 1050
    target_ratio = display_w / float(display_h)
    ratio = im_w / float(im_h)

    delta_w, delta_h = 0, 0
    if ratio > target_ratio:
        new_w = display_w
        new_h = int(new_w / ratio)
        delta_h = display_h - new_h
    else:
        new_h = display_h
        new_w = int(new_h * ratio)
        delta_w = display_w - new_w
    dif_ux = delta_w // 2
    dif_uy = delta_h // 2
    scale = im_w / float(new_w)
    X = (X - dif_ux) * scale
    Y = (Y - dif_uy) * scale
    return X, Y


def plot_scanpath(ax, img, xs, ys, ts, bbox=None, title=None, cir_rad_min=15, cir_rad_max=50):
    # fig, ax = plt.subplots()
    ax.imshow(img)
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)
    
    emlenght = min(args.emlength, len(xs))
    if args.condition == 'freeview':
      emlenght = max(args.emlength, len(xs))
    for i in range(emlenght):
        # if i == emlenght-1:
        #     plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
        #               ys[i] - ys[i - 1], width=2, color='red', alpha=0.7)
        # else:
        if i > 0:
            ax.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                    ys[i] - ys[i - 1], width=2, color='yellow', alpha=0.7)

    for i in range(emlenght):
        if i == emlenght-1:
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='red',
                                alpha=0.7)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=12, ha="center", va="center")
        else:   
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='yellow',
                                alpha=0.7)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=12, ha="center", va="center")

    # if bbox is not None:
    #     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
    #         alpha=0.7, edgecolor='yellow', facecolor='none', linewidth=2)
    #     ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    # plt.savefig('aa/scanpath'+'_'+args.condition+'_'+args.trained_model+'_'+args.task+'_'+args.imgfile)
    # plt.show()


def plot_scanpath_human(ax, img, xs, ys, ts, bbox=None, title=None, cir_rad_min=15, cir_rad_max=50, subject=1):
    # fig, ax = plt.subplots()
    ax.imshow(img)
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)
    
    emlenght = min(args.emlength, len(xs))
    if args.condition == 'freeview':
      emlenght = min(args.emlength, len(xs))
    for i in range(emlenght):
        # if i == emlenght-1:
        #     plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
        #               ys[i] - ys[i - 1], width=2, color='red', alpha=0.7)
        # else:
        if i > 0:
            ax.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                    ys[i] - ys[i - 1], width=2, color='yellow', alpha=0.7)

    for i in range(emlenght):
        if i == emlenght-1:
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='red',
                                alpha=0.7)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=12, ha="center", va="center")
        else:   
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='yellow',
                                alpha=0.7)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=12, ha="center", va="center")

    # if bbox is not None:
    #     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
    #         alpha=0.7, edgecolor='yellow', facecolor='none', linewidth=2)
    #     ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        # ax.set_title(title)
        ax.set_title( f"Subject: {scanpath['subject']}")
        
    # plt.savefig('aa/scanpath'+'_human'+str(subject)+'_'+args.condition+'_'+args.task+'_'+args.imgfile)
    # plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixation_path', type=str, help='the path of the fixation json file')
    parser.add_argument('--image_dir', type=str, help='the directory of the image stimuli')
    parser.add_argument('--random_trial', choices=[0, 1],
                        default=1, type=int, help='randomly drawn from data (default=1)')
    parser.add_argument('--trial_id', default=0, type=int, help='trial id (default=0)')
    parser.add_argument('--subj_id', type=int, default=-1,
                        help='subject id (default=-1)')
    parser.add_argument('--task', 
                        choices=['bottle', 'chair', 'cup', 'fork', 'bowl', 'mouse',
                        'microwave', 'laptop', 'key', 'sink', 'toilet', 'clock', 'tv',
                        'stop_sign', 'car', 'oven', 'knife'],
                        default='bottle',
                        help='searching target')
    parser.add_argument('--imgfile', default='stop_sign',type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args = parse_args()

    # image_dir = r'/zjh/data/FastGaze-cocosearch/images'
    # fixation_path = r'G:\scanpath_data\COCOSearch18-origin\COCOSearch18-fixations-TP\coco_search18_fixations_TP_validation_split1.json'

    # # load fixations data
    # with open(fixation_path, 'r') as f:
    #     scanpaths = json.load(f)
    # scanpaths = list(filter(lambda x: x['task'] == 'bottle', scanpaths))

    # # if args.random_trial == 1:
    # #     id = np.random.randint(len(scanpaths))
    # # else:
    # #     id = args.trial_id
    # id = 2
    # scanpath = scanpaths[id]
    # img_name = scanpath['name']
    # cat_name = scanpath['task']
    # bbox = scanpath['bbox']
    # img_path = '{}/{}/{}'.format(image_dir, cat_name, img_name)
    # print('img path', img_path)
    # print("This is target-present trial")
    # if not isfile(img_path):
    #     print("image not found at {}".format(img_path))
    #     exit(-1)
    # # load image
    # print(img_path)
    # img = mpimg.imread(img_path)
    # im_h, im_w = img.shape[0], img.shape[1]
    # # convert fixations from display coordinate to pixel coordinate
    # X, Y, T = scanpath['X'], scanpath['Y'], scanpath['T']
    # # X, Y = convert_coordinate(X, Y, im_w, im_h)
    # title = "Ground-Truth target={}".format(cat_name)
    # # plot_scanpath
    # plot_scanpath(img, X, Y, T, bbox, title)
    # print(scanpath)
    
    # 画预测的
    # cat_name = 'bottle'
    
    parser = argparse.ArgumentParser('FastGaze Test', parents=[get_args_parser_test()])
    parser.add_argument('--imgfile', default='stop_sign',type=str)
    parser.add_argument('--emlength', default=7, type=int)
    parser.add_argument('--nptask', type=str)
    
    args = parser.parse_args()
    cat_name = args.task
    
    if args.trained_model == 'FastGazeT':
       model_path = './model_zoo/fastgaze-T.pkg'
    if args.trained_model == 'FastGazeS':
       model_path = './model_zoo/fastgaze-S.pkg'
    if args.trained_model == 'FastGazeB':
       model_path = './model_zoo/fastgaze-B.pkg'
    
    num_rows = 1
    num_cols = 1  # 计算列数，确保子图不会溢出

    image_ftrs = torch.load('/zjh/data/FastGaze-cocosearch/image_features/'+args.nptask+'/'+args.imgfile[:-4]+'.pth').unsqueeze(0)
    
    for condition_asp in ['present', 'absent', 'freeview']:
        fig, axes = plt.subplots(num_rows, num_cols,dpi=300)  # 3行，根据预测数目生成列数
        args.condition = condition_asp
        predictions = test(args,image_ftrs)
        scanpath = predictions[0]
        cat_name = args.task
        
        X, Y, T = scanpath['X'], scanpath['Y'], scanpath['T']
        title = "Predicted: target={}".format(cat_name)
        print(scanpath)
        from PIL import Image
        import matplotlib.image as mpimg
        # Load the image using mpimg
        img = mpimg.imread(args.imgfile)
        h,w,c = img.shape
        print(img.shape)
        # Convert the image to a PIL image
        img_pil = Image.fromarray(img)
        # Resize the image to 320x512
        new_size = (512, 320)
        img_resized = img_pil.resize(new_size)
        img_resized = np.array(img_resized)
        width_scale = new_size[0] / w
        height_scale = new_size[1] / h
        # Resize the bounding box coordinates
        bbox = [370, 65, 187, 567]
        x_min, y_min, x_max, y_max = bbox
        new_bbox = (
            int(x_min * width_scale),
            int(y_min * height_scale),
            int(x_max * width_scale),
            int(y_max * height_scale)
        )
        new_bbox = (112, 19, 56, 172)
        T = T * width_scale
        plot_scanpath(axes, img_resized, X, Y, T, new_bbox, title, cir_rad_min=8, cir_rad_max=10)

        plt.savefig('asp_scanpath'+'_'+args.trained_model+'_'+args.condition+'_'+args.task+'_'+args.imgfile)
        plt.close()
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

def run_model(model, src, task, device="cuda:4", im_h=20, im_w=32, patch_size=16, num_samples=1):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h // 2) * patch_size, (im_w // 2) * patch_size]).unsqueeze(0).repeat(num_samples, 1)
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src=src, tgt=firstfix, task=task)
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


def test(args):

    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    vs = VS(num_encoder_layers=args.num_encoder, nhead=args.nhead, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, sc_mask=args.sc_mask, sc_ior=args.sc_ior).to(device)
    model = fastgaze(transformer=vs, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len,
                       device=device).to(device)
    print(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
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
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    task_name = args.task
    # for target_traj in test_task_img_pairs:
    task_emb = embedding_dict[task_name]
    image_ftrs = torch.load('/zjh/data/FastGaze-cocosearch/image_features/'+args.task+'/'+args.imgfile[:-4]+'.pth').unsqueeze(0)
    scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)
    for idx, scanpath in enumerate(scanpaths):
        pred_list.append((task_name, args.imgfile, args.condition, idx + 1, scanpath))
    predictions = postprocessScanpaths(pred_list)
    return predictions


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
    
    for i in range(emlenght):
        # if i == emlenght-1:
        #     plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
        #               ys[i] - ys[i - 1], width=2, color='red', alpha=0.5)
        # else:
        if i > 0:
            ax.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                    ys[i] - ys[i - 1], width=2, color='yellow', alpha=0.5)

    for i in range(emlenght):
        if i == emlenght-1:
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='red',
                                alpha=0.5)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")
        else:   
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='yellow',
                                alpha=0.5)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    # if bbox is not None:
    #     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
    #         alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
    #     ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.savefig('scanpath'+'_'+args.trained_model+'_'+args.task+'_'+args.imgfile)
    # plt.show()



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
        # fixation_path = join(dataset_root, 'coco_search18_fixations_TA_train.json')
        
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
    
    return test_target_trajs


def plot_scanpath_human(ax, img, xs, ys, ts, bbox=None, title=None, cir_rad_min=15, cir_rad_max=50, subject=1):
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
        #               ys[i] - ys[i - 1], width=2, color='red', alpha=0.5)
        # else:
        if i > 0:
            ax.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                    ys[i] - ys[i - 1], width=2, color='yellow', alpha=0.5)

    for i in range(emlenght):
        if i == emlenght-1:
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='red',
                                alpha=0.5)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")
        else:   
            cir_rad = int((cir_rad_min+cir_rad_max)/1.5 + rad_per_T * (ts[i] - min_T))
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad,
                                edgecolor='yellow',
                                facecolor='yellow',
                                alpha=0.5)
            ax.add_patch(circle)
            ax.annotate("{}".format(
                i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    ax.axis('off')
    if title is not None:
        # ax.set_title(title)
        ax.set_title( f"Subject: {scanpath['subject']}")

    # ax.text(10, 20, f"Subject: {scanpath['subject']}", fontsize=10, color='white')
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
   
    parser = argparse.ArgumentParser('FastGaze Test', parents=[get_args_parser_test()])
    parser.add_argument('--imgfile', default='stop_sign',type=str)
    parser.add_argument('--emlength', default=7, type=int)
    args = parser.parse_args()
    cat_name = args.task
    
    
    if args.trained_model == 'FastGazeB':
        model_path = './model_zoo/fastgaze-B.pkg'
    if args.trained_model == 'FastGazeS':
        model_path = './model_zoo/fastgaze-S.pkg'
    if args.trained_model == 'FastGazeT':
        model_path = './model_zoo/fastgaze-T.pkg'
        
    # human
    predictions = gethuman(args)
    
    num_rows = 3
    num_cols = (len(predictions) + num_rows - 1) // num_rows  # 计算列数，确保子图不会溢出
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 6 * num_rows),dpi=300)  # 3行，根据预测数目生成列数

    # 如果只有1个图像，确保axes变成一个二维数组
    axes = axes.flatten()  # 将axes展平成一维列表，方便循环访问
    
    for i, scanpath in enumerate(predictions):
        X, Y, T = np.array(scanpath['X']), np.array(scanpath['Y']), np.array(scanpath['T'])
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
        ax = axes[i]
        plot_scanpath_human(ax, img_resized, X, Y, T, new_bbox, title, cir_rad_min=5, cir_rad_max=8, subject=scanpath['subject'])
    
    
    predictions = test(args)
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
    plot_scanpath(axes[-1], img_resized, X, Y, T, new_bbox, title, cir_rad_min=5, cir_rad_max=8)

    plt.savefig('scanpath'+'_'+args.condition+'_'+args.task+'_'+args.imgfile)
    
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader, get_fast_dataloader_c
from utils.utils import *
from utils.imageneta import *
from utils.imagenetr import *

# attack loader
from attack.fastattack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='imagenet', type=str, help='s/a/c/r')
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=64, type=float)
parser.add_argument('--gpu', default='1', type=str)

# transformer parameter
parser.add_argument('--tran_type', default='base', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')

args = parser.parse_args()

# print configuration
print_configuration(args, 0)

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.set_device(f'cuda:{args.gpu}')

# upsampling for transformer
upsample = True if args.network in transformer_list else False

# init dataloader
if not args.dataset == 'imagenet-c':
    _, testloader, _ = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, dist=False, upsample=upsample)

# init model
net = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                  img_size=args.img_resize, patch_size=args.patch_size, pretrain=False)
net = net.cuda()

# checkpoint base tag
base_tag = '' if args.base == 'standard' else '_' + args.base

# setting checkpoint name
if args.network in transformer_list:
    net_checkpoint_name = f'checkpoint/{args.base}/imagenet/imagenet{base_tag}_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
else:
    net_checkpoint_name = f'checkpoint/{args.base}/imagenet/imagenet{base_tag}_{args.network}{args.depth}_best.t7'

# load checkpoint
net_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']

print(f"[*] Loaded network params : {net_checkpoint_name.split('/')[-1]}")
checkpoint_module(net_checkpoint, net)

def arn_test():
    net.eval()

    total = 0
    correct = 0

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        with autocast():
            cln_output = net(inputs)
            if args.dataset == 'imagenet-a':
                cln_output = cln_output[:, indices_in_1k]
            elif args.dataset == 'imagenet-r':
                cln_output = cln_output[:, imagenet_r_mask]

        _, cln_predicted = cln_output.max(1)

        total += targets.size(0)
        correct += cln_predicted.eq(targets).sum().item()

        desc = (f'[Test] Clean: {100. * correct / total:.2f}%')
        prog_bar.set_description(desc, refresh=True)

def c_test():
    net.eval()
    total_acc = 0
    total_num_cat = 18 * 5

    total_category = {
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'digital': ['contrast', 'elastic_transform', 'pixelate'],
        'extra': ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise'],
        'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
        'weather': ['brightness', 'fog', 'frost', 'snow']
    }

    for cat in total_category.keys():
        for subcat in total_category[cat]:
            for intense in range(1, 6):
                _, testloader, _ = get_fast_dataloader_c(test_batch_size=args.batch_size, dist=False, category=cat, sub_category=subcat, degree_number=str(intense))
                total = 0
                correct = 0

                prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

                for batch_idx, (inputs, targets) in prog_bar:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        cln_output = net(inputs)

                    _, cln_predicted = cln_output.max(1)

                    total += targets.size(0)
                    correct += cln_predicted.eq(targets).sum().item()

                    desc = (f'[{cat}/{subcat}/{intense}] Acc: {100. * correct / total:.2f}%')
                    prog_bar.set_description(desc, refresh=True)

                total_acc += 100. * correct / total

    print(" [*] Total Average ACC: %.2f%%" %(total_acc / total_num_cat))


if __name__ == '__main__':
    if args.dataset == 'imagenet-c':
        c_test()
    else:
        arn_test()


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
from attack.fastattack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--gpu', default='1', type=str)

# transformer parameter
parser.add_argument('--tran_type', default='small', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')

# attack parameters
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=10, type=int)

args = parser.parse_args()

# print configuration
print_configuration(args, 0)

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.set_device(f'cuda:{args.gpu}')

# upsampling for transformer
upsample = True if args.network in transformer_list else False

# init dataloader
_, testloader, _ = get_fast_dataloader(dataset=args.dataset, train_batch_size=1, test_batch_size=args.batch_size, dist=False, upsample=upsample)

# init model F
netF = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                  img_size=args.img_resize, patch_size=args.patch_size, pretrain=False)
netF = netF.cuda()

# init model G
netG = get_network(network=args.network, depth=args.depth, dataset=args.dataset, tran_type=args.tran_type,
                  img_size=args.img_resize, patch_size=args.patch_size, pretrain=False)
netG = netG.cuda()

# setting checkpoint name
if args.network in ['vit', 'deit']:
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_%d_best.t7' % (args.dataset, args.dataset,
                                                                                    args.network, args.tran_type,
                                                                                    args.patch_size, args.img_resize)
elif args.network == 'swin':
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s_daml_%s_%s_patch%d_window7_%d_best.t7' % (args.dataset, args.dataset,
                                                                                            args.network, args.tran_type,
                                                                                            args.patch_size, args.img_resize)
else:
    net_checkpoint_name = 'checkpoint/pretrain/%s/%s_daml_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)

# load checkpoint
netF_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['net']
netG_checkpoint = torch.load(net_checkpoint_name, map_location=lambda storage, loc: storage.cuda())['netG']
checkpoint_module(netF_checkpoint, netF)
checkpoint_module(netG_checkpoint, netG)
print(" [*] Loaded network params : %s" %(net_checkpoint_name.split('/')[-1]))

# init criterion
criterion = nn.CrossEntropyLoss()


def daml_test():
    netF.eval()
    netG.eval()

    attack_module = {}
    # for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
    for attack_name in ['pgd']:
        args.attack = attack_name
        if args.dataset == 'imagenet':
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps/4, steps=args.steps)
        elif args.dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps/2, steps=args.steps)
        else:
            attack_module[attack_name] = attack_loader(net=netF, attack=args.attack, eps=args.eps, steps=args.steps)

    for key in attack_module:
        total = 0
        correctF = 0
        adv_correctF = 0
        correctG = 0
        correctFG = 0
        correct_want1 = 0
        correct_want2 = 0
        correct_want3 = 0
        correct_want4 = 0
        correct_want5 = 0
        correct_want6 = 0
        correct_want7 = 0
        correct_want8 = 0
        correct_want9 = 0
        correct_want10 = 0
        correct_want11 = 0
        correct_want12 = 0
        correct_want13 = 0
        correct_want14 = 0
        correct_want15 = 0
        correct_want16 = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            with autocast():
                adv_outputsF = netF(adv_inputs)
                outputsF = netF(inputs)
                outputsG = netG(inputs)

            _, adv_predictedF = adv_outputsF.max(1)
            _, predictedF = outputsF.max(1)
            _, predictedG = (outputsF+outputsG).max(1)

            want1 = (predictedF == targets) * (adv_predictedF == targets)
            want2 = (predictedF != targets) * (adv_predictedF == targets)
            want3 = (predictedF == targets) * (adv_predictedF != targets)
            want4 = (predictedF != targets) * (adv_predictedF != targets)

            want5 = want1 * (adv_predictedF == predictedG)
            want6 = want1 * (adv_predictedF != predictedG)
            want7 = want3 * (adv_predictedF == predictedG)
            want8 = want3 * (adv_predictedF != predictedG)
            want9 = want4 * (adv_predictedF == predictedG)
            want10 = want4 * (adv_predictedF != predictedG)

            want11=want1*(predictedG==targets)
            want12=want1*(predictedG!=targets)
            want13=want3*(predictedG==targets)
            want14=want3*(predictedG!=targets)
            want15=want4*(predictedG==targets)
            want16=want4*(predictedG!=targets)

            total += targets.size(0)
            correctF += predictedF.eq(targets).sum().item()
            adv_correctF += adv_predictedF.eq(targets).sum().item()
            correctG += predictedG.eq(targets).sum().item()
            correctFG += predictedG.eq(adv_predictedF).sum().item()

            correct_want1 += want1.sum().item()
            correct_want2 += want2.sum().item()
            correct_want3 += want3.sum().item()
            correct_want4 += want4.sum().item()
            correct_want5 += want5.sum().item()
            correct_want6 += want6.sum().item()
            correct_want7 += want7.sum().item()
            correct_want8 += want8.sum().item()
            correct_want9 += want9.sum().item()
            correct_want10 += want10.sum().item()
            correct_want11 += want11.sum().item()
            correct_want12 += want12.sum().item()
            correct_want13 += want13.sum().item()
            correct_want14 += want14.sum().item()
            correct_want15 += want15.sum().item()
            correct_want16 += want16.sum().item()


            desc = (f'[Test/{key}] Clean: {100.*correctF/total:.2f}% | Adv: {100.*adv_correctF/total:.2f}% | G ACC: {100.*correctG/total:.2f}% | FG similarity: {100.*correctFG/total:.2f}%')
            prog_bar.set_description(desc, refresh=True)

        print('------------------------')
        print('(predictedF == targets) * (adv_predictedF == targets)')
        print(100. * correct_want1 / total)
        print('(predictedF != targets) * (adv_predictedF == targets)')
        print(100. * correct_want2 / total)
        print('(predictedF == targets) * (adv_predictedF != targets)')
        print(100. * correct_want3 / total)
        print('(predictedF != targets) * (adv_predictedF != targets)')
        print(100. * correct_want4 / total)
        print('------------------------')
        print('(adv_predictedF == predictedG)+(adv_predictedF != predictedG)=(predictedG==targets)+(predictedG!=targets)')
        print(f'{100. * correct_want1 / total}={100. * correct_want5 / total}+{100. * correct_want6 / total}={100. * correct_want11 / total}+{100. * correct_want12 / total}')
        print('------------------------')
        print(f'{100. * correct_want3 / total}={100. * correct_want7 / total}+{100. * correct_want8 / total}={100. * correct_want13 / total}+{100. * correct_want14 / total}')
        print('------------------------')
        print(f'{100. * correct_want4 / total}={100. * correct_want9 / total}+{100. * correct_want10 / total}={100. * correct_want15 / total}+{100. * correct_want16 / total}')
        print('------------------------')

        # fast eval
        if args.dataset == 'imagenet':
            if batch_idx >= int(len(testloader) * 0.2):
                break


def class_num(dataset_name):
    if dataset_name == 'cifar10':
        return 10, 1000
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'tiny':
        return 200, 50
    elif dataset_name == 'imagenet':
        return 1000, 50
    else:
        raise ValueError

def gen_drift_matrix(class_num, pred, targets, conf=False):
    confindence, predicted = pred.max(1)

    drift_matrix = torch.zeros([class_num[0], class_num[0]])
    conf_matrix = torch.zeros([class_num[0], class_num[0]])
    if conf:
        for index in range(predicted.shape[0]):
            drift_matrix[targets[index], predicted[index]] += 1
            conf_matrix[targets[index], predicted[index]] += confindence[index].cpu().detach()

        out_matrix = conf_matrix / drift_matrix

    else:
        for index in range(predicted.shape[0]):
            drift_matrix[targets[index], predicted[index]] += 1

        out_matrix = drift_matrix
    return out_matrix

def measure_adversarial_drift():
    netF.eval()

    attack_module = {}
    for attack_name in ['pgd']:
        args.attack = attack_name
        if args.dataset == 'imagenet':
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=args.eps/4, steps=args.steps)
        elif args.dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=args.eps/2, steps=args.steps)
        else:
            attack_module[attack_name] = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps)

    # drift matrix initialization
    drift_matrix = torch.eye(class_num(args.dataset)[0]).fill_(0)
    pred_matrix = torch.eye(class_num(args.dataset)[0]).fill_(0)

    for key in attack_module:
        total = 0
        adv_correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            adv_inputs = attack_module[key](inputs, targets)

            adv_output = netF(adv_inputs)
            _, adv_predicted = adv_output.max(1)

            # drift matrix update
            #drift_matrix += gen_drift_matrix(class_num(args.dataset), adv_output, targets)
            pred_matrix += gen_drift_matrix(class_num(args.dataset), adv_output, targets, conf=True)

            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Adv: %.4f%%' % (key, 100. * adv_correct / total))
            prog_bar.set_description(desc, refresh=True)

        pred_matrix = pred_matrix / (batch_idx + 1)

        print("ok")

if __name__ == '__main__':
    daml_test()
    #measure_adversarial_drift()



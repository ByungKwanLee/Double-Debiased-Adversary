# Future
from __future__ import print_function

# warning ignore
import warnings
warnings.filterwarnings("ignore")

import argparse

from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# fetch args
parser = argparse.ArgumentParser()

attack_list = ['plain', 'bim', 'pgd', 'cw_linf', 'ap', 'dlr', 'fab', 'aa']

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--base', default='adv', type=str)
parser.add_argument('--batch_size', default=256, type=float)

parser.add_argument('--gpu', default='0', type=str) # necessarily one gpu id!!!!

# transformer parameter
parser.add_argument('--tran_type', default='base', type=str, help='tiny/small/base/large/huge')
parser.add_argument('--img_resize', default=224, type=int, help='default/224/384')
parser.add_argument('--patch_size', default=16, type=int, help='4/16/32')

# attack parameters
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--steps', default=30, type=int)
args = parser.parse_args()


def main_worker():

    # print configuration
    print_configuration(args, 0)

    # setting gpu id of this process
    torch.cuda.set_device(f'cuda:{args.gpu}')


    # init model
    net = get_network(network=args.network,
                      depth=args.depth,
                      dataset=args.dataset,
                      tran_type=args.tran_type,
                      img_size=args.img_resize,
                      patch_size=args.patch_size,
                      pretrain=False).cuda()
    net.eval()

    # upsampling for transformer
    upsample = True if args.network in transformer_list else False

    # init dataloader
    _, testloader, _ = get_fast_dataloader(dataset=args.dataset,
                                        train_batch_size=1,
                                        test_batch_size=args.batch_size,
                                        dist=False,
                                        upsample=upsample)

    # checkpoint base tag
    base_tag = '' if args.base == 'standard' else '_' + args.base

    # setting checkpoint name
    if args.network in transformer_list:
        net_checkpoint_name = f'checkpoint/{args.base}/{args.dataset}/{args.dataset}{base_tag}_{args.network}_{args.tran_type}_patch{args.patch_size}_{args.img_resize}_best.t7'
    else:
        net_checkpoint_name = f'checkpoint/{args.base}/{args.dataset}/{args.dataset}{base_tag}_{args.network}{args.depth}_best.t7'

    rprint("This test : {}".format(net_checkpoint_name), 0)
    checkpoint = torch.load(net_checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
    checkpoint_module(checkpoint['net'], net)

    # test
    test_whitebox(net, args.dataset, testloader, attack_list=attack_list, steps=args.steps, eps=args.eps, rank=0)

if __name__ == '__main__':
    main_worker()







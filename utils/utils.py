import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict

transformer_list = ['vit', 'deit', 'swin', 'cait', 'tnt']

def do_freeze(net):
    for params in net.parameters():
        params.requires_grad = False

def rprint(str, rank):
    if rank==0:
        print(str)

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def zero_out(input):
    # return input[(input ** 2).sum(dim=-1) != 0]
    return input[input != 0]


def checkpoint_module(checkpoint, net):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_pseudo(adv_output):
    idx = adv_output.argmax(dim=1)
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, idx.unsqueeze(-1), 1)

    return psuedo_onehot

def get_onehot(adv_output, targets):
    b, c = adv_output.shape
    psuedo_onehot = torch.FloatTensor(b, c).cuda()
    psuedo_onehot.zero_()
    psuedo_onehot.scatter_(1, targets.unsqueeze(-1), 1)

    return psuedo_onehot


class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Soft cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.shape[1]
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes)
        target = (1. - self.smoothing) * target + self.smoothing / num_classes
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        loss = - (target * logprobs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class SoftCrossEntropyLoss(torch.nn.Module):
    """
    Soft cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.shape[1]
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes)
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        loss = - (target.softmax(dim=1) * logprobs)
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        return loss



def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return min_res

    if epoch >= end_ramp:
        return max_res

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32

    return final_res

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def imshow(img, norm=False):
    img = img.cpu().numpy()
    plt.imshow(np.transpose(np.array(img / 255 if norm else img, dtype=np.float32), (1, 2, 0)))
    plt.show()

def pl(a):
    plt.plot(a.cpu())
    plt.show()

def sc(a):
    plt.scatter(range(len(a.cpu())), a.cpu(), s=2, color='darkred', alpha=0.5)
    plt.show()

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def print_configuration(args, rank):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys():
            print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')

def KLDivergence(q, p):
    kld = q * (q / p).log()
    return kld.sum(dim=1)

def kld_loss(q, p):
    kld = q * ((q + 1e-4).log() - (p + 1e-4).log())
    return kld.sum(dim=1).mean()


# imagenet-c dataset loadername
# import torchvision
# # blur
# defocus_blur__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/defocus_blur/1')
# defocus_blur__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/defocus_blur/2')
# defocus_blur__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/defocus_blur/3')
# defocus_blur__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/defocus_blur/4')
# defocus_blur__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/defocus_blur/5')
# glass_blur__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/glass_blur/1')
# glass_blur__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/glass_blur/2')
# glass_blur__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/glass_blur/3')
# glass_blur__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/glass_blur/4')
# glass_blur__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/glass_blur/5')
# motion_blur__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/motion_blur/1')
# motion_blur__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/motion_blur/2')
# motion_blur__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/motion_blur/3')
# motion_blur__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/motion_blur/4')
# motion_blur__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/motion_blur/5')
# zoom_blur__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/zoom_blur/1')
# zoom_blur__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/zoom_blur/2')
# zoom_blur__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/zoom_blur/3')
# zoom_blur__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/zoom_blur/4')
# zoom_blur__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/blur/zoom_blur/5')
#
# # digital
# contrast__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/contrast/1')
# contrast__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/contrast/2')
# contrast__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/contrast/3')
# contrast__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/contrast/4')
# contrast__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/contrast/5')
# elastic_transform__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/elastic_transform/1')
# elastic_transform__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/elastic_transform/2')
# elastic_transform__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/elastic_transform/3')
# elastic_transform__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/elastic_transform/4')
# elastic_transform__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/elastic_transform/5')
# pixelate__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/pixelate/1')
# pixelate__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/pixelate/2')
# pixelate__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/pixelate/3')
# pixelate__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/pixelate/4')
# pixelate__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/digital/pixelate/5')
#
# # extra
# gaussian_blur__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/gaussian_blur/1')
# gaussian_blur__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/gaussian_blur/2')
# gaussian_blur__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/gaussian_blur/3')
# gaussian_blur__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/gaussian_blur/4')
# gaussian_blur__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/gaussian_blur/5')
# saturate__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/saturate/1')
# saturate__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/saturate/2')
# saturate__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/saturate/3')
# saturate__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/saturate/4')
# saturate__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/saturate/5')
# spatter__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/spatter/1')
# spatter__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/spatter/2')
# spatter__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/spatter/3')
# spatter__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/spatter/4')
# spatter__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/spatter/5')
# speckle_noise__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/speckle_noise/1')
# speckle_noise__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/speckle_noise/2')
# speckle_noise__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/speckle_noise/3')
# speckle_noise__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/speckle_noise/4')
# speckle_noise__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/extra/speckle_noise/5')
#
# # noise
# gaussian_noise__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/gaussian_noise/1')
# gaussian_noise__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/gaussian_noise/2')
# gaussian_noise__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/gaussian_noise/3')
# gaussian_noise__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/gaussian_noise/4')
# gaussian_noise__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/gaussian_noise/5')
# impulse_noise__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/impulse_noise/1')
# impulse_noise__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/impulse_noise/2')
# impulse_noise__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/impulse_noise/3')
# impulse_noise__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/impulse_noise/4')
# impulse_noise__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/impulse_noise/5')
# shot_noise__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/shot_noise/1')
# shot_noise__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/shot_noise/2')
# shot_noise__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/shot_noise/3')
# shot_noise__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/shot_noise/4')
# shot_noise__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/noise/shot_noise/5')
#
# # weather
# brightness__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/brightness/1')
# brightness__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/brightness/2')
# brightness__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/brightness/3')
# brightness__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/brightness/4')
# brightness__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/brightness/5')
# fog__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/fog/1')
# fog__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/fog/2')
# fog__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/fog/3')
# fog__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/fog/4')
# fog__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/fog/5')
# frost__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/frost/1')
# frost__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/frost/2')
# frost__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/frost/3')
# frost__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/frost/4')
# frost__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/frost/5')
# snow__1 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/snow/1')
# snow__2 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/snow/2')
# snow__3 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/snow/3')
# snow__4 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/snow/4')
# snow__5 = torchvision.datasets.ImageFolder('/mnt/hard1/jh_datasets/imagenet-c/weather/snow/5')

# mixattack for tiny-imagenet
class MixAttack(object):
    def __init__(self, net, slowattack, fastattack, train_iters):
        self.net = net
        self.slowattack = slowattack
        self.fastattack = fastattack
        self.train_iters = train_iters
        self.ratio = 0.3
        self.current_iter = 0

    def __call__(self, inputs, targets):
        # training
        if self.net.training:
            adv_inputs = self.slowattack(inputs, targets) \
                if self._iter < self.train_iters * self.ratio else self.fastattack(inputs, targets)
            self.iter()
            self.check()
        # testing
        else:
            adv_inputs = self.fastattack(inputs, targets)
        return adv_inputs

    def iter(self):
        self.current_iter = self.current_iter+1

    def check(self):
        if self.train_iters == self.current_iter:
            self.current_iter = 0

    @property
    def _iter(self):
        return self.current_iter


# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
from torch.cuda.amp import autocast
def test_whitebox(net, dataset, testloader, attack_list, steps, eps, rank):
    net.eval()

    attack_module = {}
    for attack_name in attack_list:
        if dataset == 'imagenet':
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps/4, steps=steps) if attack_name != 'plain' else None
        elif dataset == 'tiny':
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps/2, steps=steps) if attack_name != 'plain' else None
        else:
            attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=eps, steps=steps) if attack_name != 'plain' else None

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            if key != 'plain':
                inputs = attack_module[key](inputs, targets)
            with autocast():
                outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[White-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)

def test_blackbox(plain_net, adv_net, testloader, attack_list, eps, rank):
    plain_net.eval()
    adv_net.eval()

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=plain_net, attack=attack_name, eps=eps, steps=30)

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = attack_module[key](inputs, targets)

            with autocast():
                outputs = adv_net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Black-Box-Test/%s] Acc: %.2f%% (%d/%d)'
                    % (key, 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)

# awp package
EPS = 1E-20
def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, lr, gamma, autocast, GradScaler):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.gamma = gamma
        self.proxy_optim = torch.optim.SGD(proxy.parameters(), lr=lr)
        self.autocast = autocast
        self.scaler = GradScaler()

    def calc_awp(self, adv_inputs, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        self.proxy_optim.zero_grad()
        with self.autocast():
            loss = -F.cross_entropy(self.proxy(adv_inputs), targets)

        # Accelerating backward propagation
        self.scaler.scale(loss).backward()
        self.scaler.step(self.proxy_optim)
        self.scaler.update()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

# Causal Package
def causal_loss(logits_adv, logits_inv):
    KL = lambda x, y: (x.softmax(dim=1) * (x.softmax(dim=1).log() - y.softmax(dim=1).log())).sum(dim=1)
    return (KL(logits_adv, logits_inv)).mean()

def normalize_clip(feature, eps):
    scale = eps / torch.abs(feature).detach()
    return torch.minimum(torch.ones_like(scale), scale)
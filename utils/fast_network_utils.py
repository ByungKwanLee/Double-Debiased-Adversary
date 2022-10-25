import torch
from models.vgg import vgg
from models.resnet import resnet
from models.wide import wide_resnet
from models.vision_transformer import vit
from models.distill_transformer import deit
from models.swin_transformer import swin
from models.cattention_transformer import cait
from models.tnt_transformer import tnt
from models.tnt_transformer_b import tnt_b

def get_network(network, depth, dataset, tran_type, img_size, patch_size, pretrain):

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
        std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()
    elif dataset == 'tiny':
        mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).cuda()
        std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).cuda()
    elif dataset == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    else:
        raise NotImplementedError

    # CNN
    if network == 'vgg':
        model = vgg(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'resnet':
        model = resnet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'wide':
        model = wide_resnet(depth=depth, widen_factor=10, dataset=dataset, mean=mean, std=std)

    # Transformer
    elif network == 'vit':
        model = vit(depth=depth, vit_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                    pretrained=pretrain, mean=mean, std=std)
    elif network == 'deit':
        model = deit(depth=depth, deit_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                     pretrained=pretrain, mean=mean, std=std)
    elif network == 'swin':
        model = swin(depth=depth, swin_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                     pretrained=pretrain, mean=mean, std=std)
    elif network == 'cait':
        model = cait(depth=depth, cait_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                     pretrained=pretrain, mean=mean, std=std)
    elif network == 'tnt':
        if tran_type == 'small':
            model = tnt(depth=depth, tnt_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                         pretrained=pretrain, mean=mean, std=std)
        else:
            model = tnt_b(depth=depth, tnt_type=tran_type, img_size=img_size, patch_size=patch_size, dataset=dataset,
                          pretrained=pretrain, mean=mean, std=std)
    else:
        raise NotImplementedError

    return model
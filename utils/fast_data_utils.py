from typing import List

import torch
import torchvision
from utils.utils import *

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, RandomTranslate, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter


def save_data_for_beton(dataset, root='../data'):
    print(dataset)
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    if dataset == 'tiny':
        trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train')
        testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val')

    if dataset == 'imagenet':
        trainset = torchvision.datasets.ImageFolder('/custom_path/train')
        testset = torchvision.datasets.ImageFolder('/custom_path/val')

        # for large dataset
        datasets = {
            'train': trainset,
            'test': testset
        }
        for (name, ds) in datasets.items():

            writer = DatasetWriter(f'/custom_path/name.beton', {
                'image': RGBImageField(write_mode='jpg',
                                       max_resolution=256,
                                       compress_probability=0.5,
                                       jpeg_quality=90),
                'label': IntField(),
            }, num_workers=16)
            writer.from_indexed_dataset(ds, chunksize=100)
    else:
        # for small dataset
        datasets = {
            'train': trainset,
            'test': testset
        }
        for (name, ds) in datasets.items():
            writer = DatasetWriter(f'/custom_path/name.beton', {
                'image': RGBImageField(),
                'label': IntField(),},
                num_workers=16)
            writer.from_indexed_dataset(ds)


def get_fast_dataloader(dataset, train_batch_size, test_batch_size, num_workers=20, dist=True, shuffle=False, upsample=False):

    gpu = f'cuda:{torch.cuda.current_device()}'
    decoder = None

    if dataset == 'cifar10':
        img_size = 32
    if dataset == 'cifar100':
        img_size = 32
    if dataset == 'tiny':
        img_size = 64

    if dataset == 'imagenet':
        init_size = 160
        orgin_size = 256
        test_size = 224

        decoder = RandomResizedCropRGBImageDecoder((init_size, init_size))

        paths = {
            'train': '/custom_path',
            'test': '/custom_path'
        }
        # for large dataset
        loaders = {}
        for name in ['train', 'test']:
            if name == 'train':
                image_pipeline: List[Operation] = [decoder,
                                                   RandomHorizontalFlip()]
            else:
                image_pipeline: List[Operation] = [CenterCropRGBImageDecoder((test_size, test_size), test_size/orgin_size)]

            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze(), ToDevice_modified(torch.device(gpu), non_blocking=True)]

            image_pipeline.extend([
                ToTensor(),
                ToDevice_modified(torch.device(gpu), non_blocking=True),
                ToTorchImage(),
                Normalize_and_Convert(torch.float16, True)
            ])
            if shuffle:
                order = OrderOption.RANDOM
            else:
                order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

            loaders[name] = Loader(paths[name], batch_size=train_batch_size if name == 'train' else test_batch_size,
                                   num_workers=num_workers, order=order, drop_last=(name == 'train'), os_cache=True,
                                   distributed=dist, pipelines={'image': image_pipeline, 'label': label_pipeline},
                                   seed = 0)
    else:
        # for small dataset
        paths = {
            'train': '/custom_path',
            'test': '/custom_path'
        }

        loaders = {}
        for name in ['train', 'test']:
            if name == 'train':
                if upsample:
                    image_pipeline: List[Operation] = [CenterCropRGBImageDecoder((224,224), ratio=1),
                                                       RandomHorizontalFlip(),
                                                       RandomTranslate(padding=28)]
                else:
                    image_pipeline: List[Operation] = [SimpleRGBImageDecoder(),
                                                       RandomHorizontalFlip(),
                                                       RandomTranslate(padding=int(img_size/8.))]
            else:
                if upsample:
                    image_pipeline: List[Operation] = [CenterCropRGBImageDecoder((224,224), ratio=1)]
                else:
                    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice_modified(torch.device(gpu)), Squeeze()]

            image_pipeline.extend([
                ToTensor(),
                ToDevice_modified(torch.device(gpu), non_blocking=True),
                ToTorchImage(),
                Normalize_and_Convert(torch.float16, True)
            ])
            if shuffle:
                order = OrderOption.RANDOM
            else:
                order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

            loaders[name] = Loader(paths[name], batch_size=train_batch_size if name == 'train' else test_batch_size,
                                num_workers=num_workers, order=order, drop_last=(name == 'train'), os_cache=True,
                                   distributed=dist, pipelines={'image': image_pipeline, 'label': label_pipeline},
                                   seed=0)

    return loaders['train'], loaders['test'], decoder

# Custom Tranforms
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.state import State
from ffcv.pipeline.allocation_query import AllocationQuery

# Normalize operation to Normalize and Convert
class Normalize_and_Convert(Operation):
    def __init__(self, target_dtype, target_norm_bool):
        super().__init__()
        self.target_dtype = target_dtype
        self.target_norm_bool = target_norm_bool

    def generate_code(self) -> Callable:
        def convert(inp, dst):
            if self.target_norm_bool:
                inp = inp / 255.0
            return inp.type(self.target_dtype)
        convert.is_parallel = True
        return convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype), None

# ToDevice to be modified version
from ffcv.transforms import ToDevice
class ToDevice_modified(ToDevice):
    def __init__(self, device, non_blocking=True):
        super(ToDevice_modified, self).__init__(device, non_blocking)

    def generate_code(self):
        def to_device(inp, dst):
            if len(inp.shape) == 4:
                if inp.is_contiguous(memory_format=torch.channels_last):
                    dst = dst.reshape(inp.shape[0], inp.shape[2], inp.shape[3], inp.shape[1])
                    dst = dst.permute(0, 3, 1, 2)

            if len(inp.shape) == 0:
                inp = inp.unsqueeze(0)

            dst = dst[:inp.shape[0]]
            dst.copy_(inp, non_blocking=self.non_blocking)
            return dst

        return to_device

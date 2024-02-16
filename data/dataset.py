import random
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path


def get_dataset_path(dataset_name, height, file_suffix, datasets_path):
    if file_suffix is not None:
        filename = f'{dataset_name}-{height}-{file_suffix}.pickle'
    else:
        filename = f'{dataset_name}-{height}.pickle'

    return os.path.join(datasets_path, filename)


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class TextDataset:

    def __init__(self, base_path, collator_resolution, num_examples=15, target_transform=None, min_virtual_size=0, validation=False, debug=False):
        self.NUM_EXAMPLES = num_examples
        self.debug = debug
        self.min_virtual_size = min_virtual_size

        subset = 'test' if validation else 'train'

        # base_path=DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)[subset]
        self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [:NUM_WRITERS])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']

        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        if self.debug:
            return 16
        return max(len(self.author_id), self.min_virtual_size)

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        index = index % len(self.author_id)

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = random.choices([i for i in range(len(self.IMG_DATA_AUTHOR))], k=self.NUM_EXAMPLES)

        word_data = random.choice(self.IMG_DATA_AUTHOR)
        real_img = self.transform(word_data['img'].convert('L'))
        real_labels = word_data['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        slabels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img_height, img_width = img.shape[0], img.shape[1]
            output_img = np.ones((img_height, max_width), dtype='float32') * 255.0
            output_img[:, :img_width] = img[:, :max_width]

            imgs_pad.append(self.transform(Image.fromarray(output_img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,   # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
            'swids': imgs_wids, # widths of the N images [list(N)]
            'img': real_img,  # the input image [1, H (32), W]
            'label': real_labels,  # the label of the input image [byte]
            'img_path': 'img_path',
            'idx': 'indexes',
            'wcl': index,  # id of the author [int],
            'slabels': slabels,
            'author_id': author_id
        }
        return item

    def get_stats(self):
        char_counts = defaultdict(lambda: 0)
        total = 0

        for author in self.IMG_DATA.keys():
            for data in self.IMG_DATA[author]:
                for char in data['label']:
                    char_counts[char] += 1
                    total += 1

        char_counts = {k: 1.0 / (v / total) for k, v in char_counts.items()}

        return char_counts


class TextCollator(object):
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, batch):
        if isinstance(batch[0], list):
            batch = sum(batch, [])
        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        simgs = torch.stack([item['simg'] for item in batch], 0)
        wcls = torch.Tensor([item['wcl'] for item in batch])
        swids = torch.Tensor([item['swids'] for item in batch])
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path': img_path, 'idx': indexes, 'simg': simgs, 'swids': swids, 'wcl': wcls}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'slabels' in batch[0].keys():
            slabels = [item['slabels'] for item in batch]
            item['slabels'] = np.array(slabels)
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item


class CollectionTextDataset(Dataset):
    def __init__(self, datasets, datasets_path, dataset_class, file_suffix=None, height=32, **kwargs):
        self.datasets = {}
        for dataset_name in sorted(datasets.split(',')):
            dataset_file = get_dataset_path(dataset_name, height, file_suffix, datasets_path)
            dataset = dataset_class(dataset_file, **kwargs)
            self.datasets[dataset_name] = dataset
        self.alphabet = ''.join(sorted(set(''.join(d.alphabet for d in self.datasets.values()))))

    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    @property
    def num_writers(self):
        return sum(d.num_writers for d in self.datasets.values())

    def __getitem__(self, index):
        for dataset in self.datasets.values():
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError

    def get_dataset(self, index):
        for dataset_name, dataset in self.datasets.items():
            if index < len(dataset):
                return dataset_name
            index -= len(dataset)
        raise IndexError

    def collate_fn(self, batch):
        return self.datasets[self.get_dataset(0)].collate_fn(batch)


class FidDataset(Dataset):
    def __init__(self, base_path, collator_resolution, num_examples=15, target_transform=None, mode='train', style_dataset=None):
        self.NUM_EXAMPLES = num_examples

        # base_path=DATASET_PATHS
        with open(base_path, "rb") as f:
            self.IMG_DATA = pickle.load(f)

        self.IMG_DATA = self.IMG_DATA[mode]
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']

        self.STYLE_IMG_DATA = None
        if style_dataset is not None:
            with open(style_dataset, "rb") as f:
                self.STYLE_IMG_DATA = pickle.load(f)

                self.STYLE_IMG_DATA = self.STYLE_IMG_DATA[mode]
                if 'None' in self.STYLE_IMG_DATA.keys():
                    del self.STYLE_IMG_DATA['None']

        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = sorted(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        self.dataset_size = sum(len(samples) for samples in self.IMG_DATA.values())
        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        return self.dataset_size

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        sample, author_id = None, None
        for author_id, samples in self.IMG_DATA.items():
            if index < len(samples):
                sample, author_id = samples[index], author_id
                break
            index -= len(samples)

        real_image = self.transform(sample['img'].convert('L'))
        real_label = sample['label'].encode()

        style_dataset = self.STYLE_IMG_DATA if self.STYLE_IMG_DATA is not None else self.IMG_DATA

        author_style_images = style_dataset[author_id]
        random_idxs = np.random.choice(len(author_style_images), NUM_SAMPLES, replace=True)
        style_images = [np.array(author_style_images[idx]['img'].convert('L')) for idx in random_idxs]

        max_width = 192

        imgs_pad = []
        imgs_wids = []

        for img in style_images:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
            'img': real_image,  # the input image [1, H (32), W]
            'label': real_label,  # the label of the input image [byte]
            'img_path': 'img_path',
            'idx': sample['img_id'] if 'img_id' in sample.keys() else sample['image_id'],
            'wcl': int(author_id)  # id of the author [int]
        }
        return item


class FolderDataset:
    def __init__(self, folder_path, num_examples=15, word_lengths=None):
        folder_path = Path(folder_path)
        self.imgs = list([p for p in folder_path.iterdir() if not p.suffix == '.txt'])
        self.transform = get_transform(grayscale=True)
        self.num_examples = num_examples
        self.word_lengths = word_lengths

    def __len__(self):
        return len(self.imgs)

    def sample_style(self):
        random_idxs = np.random.choice(len(self.imgs), self.num_examples, replace=False)
        image_names = [self.imgs[idx].stem for idx in random_idxs]
        imgs = [Image.open(self.imgs[idx]).convert('L') for idx in random_idxs]
        if self.word_lengths is None:
            imgs = [img.resize((img.size[0] * 32 // img.size[1], 32), Image.BILINEAR) for img in imgs]
        else:
            imgs = [img.resize((self.word_lengths[name] * 16, 32), Image.BILINEAR) for img, name in zip(imgs, image_names)]
        imgs = [np.array(img) for img in imgs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
        }
        return item

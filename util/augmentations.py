import os.path
import pickle
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np
import math
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from data.dataset import CollectionTextDataset, TextDataset


def to_opencv(batch: torch.Tensor):
    images = []

    for image in batch:
        image = image.detach().cpu().numpy()
        image = (image + 1.0) / 2.0
        images.append(np.squeeze(image))

    return images


class RandomMorphological(torch.nn.Module):
    def __init__(self, max_size: 5, max_iterations = 1, operation = cv2.MORPH_ERODE):
        super().__init__()
        self.elements = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE]
        self.max_size = max_size
        self.max_iterations = max_iterations
        self.operation = operation

    def forward(self, x):
        device = x.device

        images = to_opencv(x)

        result = []

        size = random.randint(1, self.max_size)
        kernel = cv2.getStructuringElement(random.choice(self.elements), (size, size))

        for image in images:
            image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
            morphed = cv2.morphologyEx(image, op=self.operation, kernel=kernel, iterations=random.randint(1, self.max_iterations))
            morphed = cv2.resize(morphed, (image.shape[1] // 2, image.shape[0] // 2))
            morphed = morphed * 2.0 - 1.0

            result.append(torch.Tensor(morphed))

        return torch.unsqueeze(torch.stack(result).to(device), dim=1)


def gauss_noise_tensor(img):
    # https://github.com/pytorch/vision/issues/6192
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 0.075

    out = img + sigma * (torch.randn_like(img) - 0.5)

    out = torch.clamp(out, -1.0, 1.0)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


def compute_word_width(image: torch.Tensor) -> int:
    indices = torch.where((image < 0).int())[2]
    index = torch.max(indices) if len(indices) > 0 else image.size(-1)

    return index


class Downsize(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(0.0, scale=(0.8, 1.0), interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=1.0),
            torchvision.transforms.GaussianBlur(3, sigma=0.3)
        ])

    def forward(self, x):
        return self.aug(x)


class OCRAugment(torch.nn.Module):
    def __init__(self, prob: float = 0.5, no: int = 2):
        super().__init__()
        self.prob = prob
        self.no = no

        interp = torchvision.transforms.InterpolationMode.NEAREST
        fill = 1.0

        self.augmentations = [
            torchvision.transforms.RandomRotation(3.0, interpolation=interp, fill=fill),
            torchvision.transforms.RandomAffine(0.0, translate=(0.05, 0.05), interpolation=interp, fill=fill),
            Downsize(),
            torchvision.transforms.ElasticTransform(alpha=10.0, sigma=7.0, fill=fill, interpolation=interp),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5),
            torchvision.transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            gauss_noise_tensor,
            RandomMorphological(max_size=4, max_iterations=2, operation=cv2.MORPH_ERODE),
            RandomMorphological(max_size=2, max_iterations=1, operation=cv2.MORPH_DILATE)
        ]

    def forward(self, x):
        if random.uniform(0.0, 1.0) > self.prob:
            return x

        augmentations = random.choices(self.augmentations, k=self.no)

        for augmentation in augmentations:
            x = augmentation(x)

        return x


class WordCrop(torch.nn.Module, ABC):
    def __init__(self, use_padding: bool = False):
        super().__init__()
        self.use_padding = use_padding
        self.pad = torchvision.transforms.Pad([2, 2, 2, 2], 1.0)

    @abstractmethod
    def get_current_width(self):
        pass

    @abstractmethod
    def update(self, epoch: int):
        pass

    def forward(self, images):
        assert len(images.size()) == 4 and images.size(1) == 1, "Augmentation works on batches of one channel images"

        if self.use_padding:
            images = self.pad(images)

        results = []
        width = self.get_current_width()

        for image in images:
            index = compute_word_width(image)
            max_index = max(min(index - width // 2, image.size(2) - width), 0)
            start_index = random.randint(0, max_index)

            results.append(F.crop(image, 0, start_index, image.size(1), min(width, image.size(2))))

        return torch.stack(results)


class StaticWordCrop(WordCrop):
    def __init__(self, width: int, use_padding: bool = False):
        super().__init__(use_padding=use_padding)
        self.width = width

    def get_current_width(self):
        return int(self.width)

    def update(self, epoch: int):
        pass


class RandomWordCrop(WordCrop):
    def __init__(self, min_width: int, max_width: int, use_padding: bool = False):
        super().__init__(use_padding)

        self.min_width = min_width
        self.max_width = max_width

        self.current_width = random.randint(self.min_width, self.max_width)

    def update(self, epoch: int):
        self.current_width = random.randint(self.min_width, self.max_width)

    def get_current_width(self):
        return self.current_width


class FullCrop(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.height = 32
        self.pad = torchvision.transforms.Pad([6, 6, 6, 6], 1.0)

    def get_current_width(self):
        return self.width

    def forward(self, images):
        assert len(images.size()) == 4 and images.size(1) == 1, "Augmentation works on batches of one channel images"
        images = self.pad(images)

        results = []

        for image in images:
            index = compute_word_width(image)
            max_index = max(min(index - self.width // 2, image.size(2) - self.width), 0)

            start_width = random.randint(0, max_index)
            start_height = random.randint(0, image.size(1) - self.height)

            results.append(F.crop(image, start_height, start_width, self.height, min(self.width, image.size(2))))

        return torch.stack(results)


class ProgressiveWordCrop(WordCrop):
    def __init__(self, width: int, warmup_epochs: int, start_width: int = 128, use_padding: bool = False):
        super().__init__(use_padding=use_padding)
        self.target_width = width
        self.warmup_epochs = warmup_epochs
        self.start_width = start_width
        self.current_width = float(start_width)

    def update(self, epoch: int):
        value = self.start_width - ((self.start_width - self.target_width) / self.warmup_epochs) * epoch
        self.current_width = max(value, self.target_width)

    def get_current_width(self):
        return int(round(self.current_width))


class CycleWordCrop(WordCrop):
    def __init__(self, width: int, cycle_epochs: int, start_width: int = 128, use_padding: bool = False):
        super().__init__(use_padding=use_padding)

        self.target_width = width
        self.start_width = start_width
        self.current_width = float(start_width)
        self.cycle_epochs = float(cycle_epochs)

    def update(self, epoch: int):
        value = (math.cos((float(epoch) * 2 * math.pi) / self.cycle_epochs) + 1) * ((self.start_width - self.target_width) / 2) + self.target_width
        self.current_width = value

    def get_current_width(self):
        return int(round(self.current_width))


class HeightResize(torch.nn.Module):
    def __init__(self, target_height: int):
        super().__init__()
        self.target_height = target_height

    def forward(self, x):
        width, height = F.get_image_size(x)
        scale = self.target_height / height

        return F.resize(x, [int(height * scale), int(width * scale)])



def show_crops():
    with open("../files/IAM-32-pa.pickle", 'rb') as f:
        data = pickle.load(f)

    for author in data['train'].keys():
        for image in data['train'][author]:
            image = torch.Tensor(np.expand_dims(np.expand_dims(np.array(image['img']), 0), 0))

            augmenter = torchvision.transforms.Compose([
                HeightResize(32),
                FullCrop(128)
            ])

            batch = augmenter(image)

            batch = batch.detach().cpu().numpy()
            result = [np.squeeze(im) for im in batch]

            #plt.imshow(np.squeeze(image))

            f, ax = plt.subplots(1, len(result))

            for i in range(len(result)):
                ax.imshow(result[i])

            plt.show()


if __name__ == "__main__":
    dataset = CollectionTextDataset(
        'IAM', '../files', TextDataset, file_suffix='pa', num_examples=15,
        collator_resolution=16, min_virtual_size=339, validation=False, debug=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn)

    augmenter = OCRAugment(no=3, prob=1.0)

    target_folder = r"C:\Users\bramv\Documents\Werk\Research\Unimore\VATr\VATr_ext\saved_images\debug\ocr_aug"

    image_no = 0

    for batch in train_loader:
        for i in range(5):
            augmented = augmenter(batch["img"])

            img = np.squeeze((augmented[0].detach().cpu().numpy() + 1.0) / 2.0)

            img = (img * 255.0).astype(np.uint8)

            print(cv2.imwrite(os.path.join(target_folder, f"{image_no}_{i}.png"), img))

        img = np.squeeze((batch["img"][0].detach().cpu().numpy() + 1.0) / 2.0)
        img = (img * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(target_folder, f"{image_no}.png"), img)

        if image_no > 5:
            break

        image_no+=1


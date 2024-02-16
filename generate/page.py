import os

import cv2
import numpy as np
import torch

from data.dataset import CollectionTextDataset, TextDataset
from models.model import VATr
from util.loading import load_checkpoint, load_generator


def generate_page(args):
    args.output = 'vatr' if args.output is None else args.output

    args.vocab_size = len(args.alphabet)

    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution
    )
    datasetval = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, validation=True
    )

    args.num_writers = dataset.num_writers

    model = VATr(args)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = load_generator(model, checkpoint)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        datasetval,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True, drop_last=True,
        collate_fn=datasetval.collate_fn)

    data_train = next(iter(train_loader))
    data_val = next(iter(val_loader))

    model.eval()
    with torch.no_grad():
        page = model._generate_page(data_train['simg'].to(args.device), data_val['swids'])
        page_val = model._generate_page(data_val['simg'].to(args.device), data_val['swids'])

    cv2.imwrite(os.path.join("saved_images", "pages", f"{args.output}_train.png"), (page * 255).astype(np.uint8))
    cv2.imwrite(os.path.join("saved_images", "pages", f"{args.output}_val.png"), (page_val * 255).astype(np.uint8))

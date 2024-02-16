import os
import shutil

import cv2
import msgpack
import torch

from data.dataset import CollectionTextDataset, TextDataset, FolderDataset, FidDataset, get_dataset_path
from generate.writer import Writer
from util.text import get_generator


def generate_ocr(args):
    """
    Generate OCR training data. Words generated are from given text generator.
    """
    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, validation=True
    )
    args.num_writers = dataset.num_writers

    writer = Writer(args.checkpoint, args, only_generator=True)

    generator = get_generator(args)

    writer.generate_ocr(dataset, args.count, interpolate_style=args.interp_styles, output_folder=args.output, text_generator=generator)


def generate_ocr_reference(args):
    """
    Generate OCR training data. Words generated are words from given dataset. Reference words are also saved.
    """
    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, validation=True
    )

    #dataset = FidDataset(get_dataset_path(args.dataset, 32, args.file_suffix, 'files'), mode='test', collator_resolution=args.resolution)

    args.num_writers = dataset.num_writers

    writer = Writer(args.checkpoint, args, only_generator=True)

    writer.generate_ocr(dataset, args.count, interpolate_style=args.interp_styles, output_folder=args.output, long_tail=args.long_tail)


def generate_ocr_msgpack(args):
    """
    Generate OCR dataset. Words generated are specified in given msgpack file
    """
    dataset = FolderDataset(args.dataset_path)
    args.num_writers = 339

    if args.charset_file:
        charset = msgpack.load(open(args.charset_file, 'rb'), use_list=False, strict_map_key=False)
        args.alphabet = "".join(charset['char2idx'].keys())

    writer = Writer(args.checkpoint, args, only_generator=True)

    lines = msgpack.load(open(args.text_path, 'rb'), use_list=False)

    print(f"Generating {len(lines)} to {args.output}")

    for i, (filename, target) in enumerate(lines):
        if not os.path.exists(os.path.join(args.output, filename)):
            style = torch.unsqueeze(dataset.sample_style()['simg'], dim=0).to(args.device)
            fake = writer.create_fake_sentence(style, target, at_once=True)

            cv2.imwrite(os.path.join(args.output, filename), fake)

    print(f"Done")

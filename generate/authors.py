import os
import shutil

import cv2
import numpy as np

from data.dataset import CollectionTextDataset, TextDataset
from generate.util import stack_lines
from generate.writer import Writer


def generate_authors(args):
    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, file_suffix=args.file_suffix, num_examples=args.num_examples,
        collator_resolution=args.resolution, validation=args.test_set
    )

    args.num_writers = dataset.num_writers

    writer = Writer(args.checkpoint, args, only_generator=True)

    if args.text.endswith(".txt"):
        with open(args.text, 'r') as f:
            lines = [l.rstrip() for l in f]
    else:
        lines = [args.text]

    output_dir = "saved_images/author_samples/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    fakes, author_ids, style_images = writer.generate_authors(lines, dataset, args.align, args.at_once)

    for fake, author_id, style in zip(fakes, author_ids, style_images):
        author_dir = os.path.join(output_dir, str(author_id))
        os.mkdir(author_dir)

        for i, line in enumerate(fake):
            cv2.imwrite(os.path.join(author_dir, f"line_{i}.png"), line)

        total = stack_lines(fake)
        cv2.imwrite(os.path.join(author_dir, "total.png"), total)

        if args.output_style:
            for i, image in enumerate(style):
                cv2.imwrite(os.path.join(author_dir, f"style_{i}.png"), image)


import os
from pathlib import Path

import torch
import torch.utils.data

from data.dataset import FidDataset
from generate.writer import Writer


def generate_fid(args):
    if 'iam' in args.target_dataset_path.lower():
        args.num_writers = 339
    elif 'cvl' in args.target_dataset_path.lower():
        args.num_writers = 283
    else:
        raise ValueError

    args.vocab_size = len(args.alphabet)

    dataset_train = FidDataset(base_path=args.target_dataset_path, num_examples=args.num_examples, collator_resolution=args.resolution, mode='train', style_dataset=args.dataset_path)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
        collate_fn=dataset_train.collate_fn
    )

    dataset_test = FidDataset(base_path=args.target_dataset_path, num_examples=args.num_examples, collator_resolution=args.resolution, mode='test', style_dataset=args.dataset_path)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, drop_last=False,
        collate_fn=dataset_test.collate_fn
    )

    args.output = 'saved_images' if args.output is None else args.output
    args.output = Path(args.output) / 'fid' / args.target_dataset_path.split("/")[-1].replace(".pickle", "").replace("-", "")

    model_folder = args.checkpoint.split("/")[-2] if args.checkpoint.endswith(".pth") else args.checkpoint.split("/")[-1]
    model_tag = model_folder.split("-")[-1] if "-" in model_folder else "vatr"
    model_tag += "_" + args.dataset_path.split("/")[-1].replace(".pickle", "").replace("-", "")

    if not args.all_epochs:
        writer = Writer(args.checkpoint, args, only_generator=True)
        if not args.test_only:
            writer.generate_fid(args.output, train_loader, model_tag=model_tag, split='train', fake_only=args.fake_only, long_tail_only=args.long_tail)
        writer.generate_fid(args.output, test_loader, model_tag=model_tag, split='test', fake_only=args.fake_only, long_tail_only=args.long_tail)
    else:
        epochs = sorted([int(f.split("_")[0]) for f in os.listdir(args.checkpoint) if "_" in f])
        generate_real = True

        for epoch in epochs:
            checkpoint_path = os.path.join(args.checkpoint, f"{str(epoch).zfill(4)}_model.pth")
            writer = Writer(checkpoint_path, args, only_generator=True)
            writer.generate_fid(args.output, test_loader, model_tag=f"{model_tag}_{epoch}", split='test', fake_only=not generate_real, long_tail_only=args.long_tail)
            generate_real = False

    print('Done')

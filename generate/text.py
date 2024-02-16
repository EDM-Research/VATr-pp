from pathlib import Path

import cv2

from generate.writer import Writer


def generate_text(args):
    if args.text_path is not None:
        with open(args.text_path, 'r') as f:
            args.text = f.read()
    args.text = args.text.splitlines()
    args.output = 'files/output.png' if args.output is None else args.output
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.num_writers = 0

    writer = Writer(args.checkpoint, args, only_generator=True)
    writer.set_style_folder(args.style_folder)
    fakes = writer.generate(args.text, args.align)
    for i, fake in enumerate(fakes):
        dst_path = args.output.parent / (args.output.stem + f'_{i:03d}' + args.output.suffix)
        cv2.imwrite(str(dst_path), fake)
    print('Done')

import json
import os
import random
import shutil
from collections import defaultdict
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

from data.dataset import FolderDataset
from models.model import VATr
from util.loading import load_checkpoint, load_generator
from util.misc import FakeArgs
from util.text import TextGenerator
from util.vision import detect_text_bounds


def get_long_tail_chars():
    with open(f"files/longtail.txt", 'r') as f:
        chars = [c.rstrip() for c in f]

    chars.remove('')

    return chars


class Writer:
    def __init__(self, checkpoint_path, args, only_generator: bool = False):
        self.model = VATr(args)
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        load_checkpoint(self.model, checkpoint) if not only_generator else load_generator(self.model, checkpoint)
        self.model.eval()
        self.style_dataset = None

    def set_style_folder(self, style_folder, num_examples=15):
        word_lengths = None
        if os.path.exists(os.path.join(style_folder, "word_lengths.txt")):
            word_lengths = {}
            with open(os.path.join(style_folder, "word_lengths.txt"), 'r') as f:
                for line in f:
                    word, length = line.rstrip().split(",")
                    word_lengths[word] = int(length)

        self.style_dataset = FolderDataset(style_folder, num_examples=num_examples, word_lengths=word_lengths)

    @torch.no_grad()
    def generate(self, texts, align_words: bool = False, at_once: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        if self.style_dataset is None:
            raise Exception('Style is not set')

        fakes = []
        for i, text in enumerate(texts, 1):
            print(f'[{i}/{len(texts)}] Generating for text: {text}')
            style = self.style_dataset.sample_style()
            style_images = style['simg'].unsqueeze(0).to(self.model.args.device)

            fake = self.create_fake_sentence(style_images, text, align_words, at_once)

            fakes.append(fake)
        return fakes

    @torch.no_grad()
    def create_fake_sentence(self, style_images, text, align_words=False, at_once=False):
        text = "".join([c for c in text if c in self.model.args.alphabet])

        text = text.split() if not at_once else [text]
        gap = np.ones((32, 16))

        text_encode, len_text, encode_pos = self.model.netconverter.encode(text)
        text_encode = text_encode.to(self.model.args.device).unsqueeze(0)

        fake = self.model._generate_fakes(style_images, text_encode, len_text)
        if not at_once:
            if align_words:
                fake = self.stitch_words(fake, show_lines=False)
            else:
                fake = np.concatenate(sum([[img, gap] for img in fake], []), axis=1)[:, :-16]
        else:
            fake = fake[0]
        fake = (fake * 255).astype(np.uint8)

        return fake

    @torch.no_grad()
    def generate_authors(self, text, dataset, align_words: bool = False, at_once: bool = False):
        fakes = []
        author_ids = []
        style = []

        for item in dataset:
            print(f"Generating author {item['wcl']}")
            style_images = item['simg'].to(self.model.args.device).unsqueeze(0)

            generated_lines = [self.create_fake_sentence(style_images, line, align_words, at_once) for line in text]

            fakes.append(generated_lines)
            author_ids.append(item['author_id'])
            style.append((((item['simg'].numpy() + 1.0) / 2.0) * 255).astype(np.uint8))

        return fakes, author_ids, style

    @torch.no_grad()
    def generate_characters(self, dataset, characters: str):
        """
        Generate each of the given characters for each of the authors in the dataset.
        """
        fakes = []

        text_encode, len_text, encode_pos = self.model.netconverter.encode([c for c in characters])
        text_encode = text_encode.to(self.model.args.device).unsqueeze(0)

        for item in dataset:
            print(f"Generating author {item['wcl']}")
            style_images = item['simg'].to(self.model.args.device).unsqueeze(0)
            fake = self.model.netG.evaluate(style_images, text_encode)

            fakes.append(fake)

        return fakes

    @torch.no_grad()
    def generate_batch(self, style_imgs, text):
        """
        Given a batch of style images and text, generate images using the model
        """
        device = self.model.args.device
        text_encode, _, _ = self.model.netconverter.encode(text)
        fakes, _ = self.model.netG(style_imgs.to(device), text_encode.to(device))
        return fakes

    @torch.no_grad()
    def generate_ocr(self, dataset, number: int, output_folder: str = 'saved_images/ocr', interpolate_style: bool = False, text_generator: TextGenerator = None, long_tail: bool = False):
        def create_and_write(style, text, interpolated=False):
            nonlocal image_counter, annotations

            text_encode, len_text, encode_pos = self.model.netconverter.encode([text])
            text_encode = text_encode.to(self.model.args.device)

            fake = self.model.netG.generate(style, text_encode)

            fake = (fake + 1) / 2
            fake = fake.cpu().numpy()
            fake = np.squeeze((fake * 255).astype(np.uint8))

            image_filename = f"{image_counter}.png" if not interpolated else f"{image_counter}_i.png"

            cv2.imwrite(os.path.join(output_folder, "generated", image_filename), fake)

            annotations.append((image_filename, text))

            image_counter += 1

        image_counter = 0
        annotations = []
        previous_style = None
        long_tail_chars = get_long_tail_chars()

        os.mkdir(os.path.join(output_folder, "generated"))
        if text_generator is None:
            os.mkdir(os.path.join(output_folder, "reference"))

        while image_counter < number:
            author_index = random.randint(0, len(dataset) - 1)
            item = dataset[author_index]

            style_images = item['simg'].to(self.model.args.device).unsqueeze(0)
            style = self.model.netG.compute_style(style_images)

            if interpolate_style and previous_style is not None:
                factor = float(np.clip(random.gauss(0.5, 0.15), 0.0, 1.0))
                intermediate_style = torch.lerp(previous_style, style, factor)
                text = text_generator.generate()

                create_and_write(intermediate_style, text, interpolated=True)

            if text_generator is not None:
                text = text_generator.generate()
            else:
                text = str(item['label'].decode())

                if long_tail and not any(c in long_tail_chars for c in text):
                    continue

                fake = (item['img'] + 1) / 2
                fake = fake.cpu().numpy()
                fake = np.squeeze((fake * 255).astype(np.uint8))

                image_filename = f"{image_counter}.png"

                cv2.imwrite(os.path.join(output_folder, "reference", image_filename), fake)

            create_and_write(style, text)

            previous_style = style

        if text_generator is None:
            with open(os.path.join(output_folder, "reference", "labels.csv"), 'w') as fr:
                fr.write(f"filename,words\n")
                for annotation in annotations:
                    fr.write(f"{annotation[0]},{annotation[1]}\n")

        with open(os.path.join(output_folder, "generated", "labels.csv"), 'w') as fg:
            fg.write(f"filename,words\n")
            for annotation in annotations:
                fg.write(f"{annotation[0]},{annotation[1]}\n")


    @staticmethod
    def stitch_words(words: list, show_lines: bool = False, scale_words: bool = False):
        gap_width = 16

        bottom_lines = []
        top_lines = []
        for i in range(len(words)):
            b, t = detect_text_bounds(words[i])
            bottom_lines.append(b)
            top_lines.append(t)
            if show_lines:
                words[i] = cv2.line(words[i], (0, b), (words[i].shape[1], b), (0, 0, 1.0))
                words[i] = cv2.line(words[i], (0, t), (words[i].shape[1], t), (1.0, 0, 0))

        bottom_lines = np.array(bottom_lines, dtype=float)

        if scale_words:
            top_lines = np.array(top_lines, dtype=float)
            gaps = bottom_lines - top_lines
            target_gap = np.mean(gaps)
            scales = target_gap / gaps

            bottom_lines *= scales
            top_lines *= scales
            words = [cv2.resize(word, None, fx=scale, fy=scale) for word, scale in zip(words, scales)]

        highest = np.max(bottom_lines)
        offsets = highest - bottom_lines
        height = np.max(offsets + [word.shape[0] for word in words])

        result = np.ones((int(height), gap_width * len(words) + sum([w.shape[1] for w in words])))

        x_pos = 0
        for bottom_line, word in zip(bottom_lines, words):
            offset = int(highest - bottom_line)

            result[offset:offset + word.shape[0], x_pos:x_pos+word.shape[1]] = word

            x_pos += word.shape[1] + gap_width

        return result

    @torch.no_grad()
    def generate_fid(self, path, loader, model_tag, split='train', fake_only=False, long_tail_only=False):
        if not isinstance(path, Path):
            path = Path(path)

        path.mkdir(exist_ok=True, parents=True)

        appendix = f"{split}" if not long_tail_only else f"{split}_lt"

        real_base = path / f'real_{appendix}'
        fake_base = path / model_tag / f'fake_{appendix}'

        if real_base.exists() and not fake_only:
            shutil.rmtree(real_base)

        if fake_base.exists():
            shutil.rmtree(fake_base)

        real_base.mkdir(exist_ok=True)
        fake_base.mkdir(exist_ok=True, parents=True)

        print('Saving images...')

        print('  Saving images on {}'.format(str(real_base)))
        print('  Saving images on {}'.format(str(fake_base)))

        long_tail_chars = get_long_tail_chars()
        counter = 0
        ann = defaultdict(lambda: {})
        start_time = time.time()
        for step, data in enumerate(loader):
            style_images = data['simg'].to(self.model.args.device)

            texts = [l.decode('utf-8') for l in data['label']]
            texts = [t.encode('utf-8') for t in texts]
            eval_text_encode, eval_len_text, _ = self.model.netconverter.encode(texts)
            eval_text_encode = eval_text_encode.to(self.model.args.device).unsqueeze(1)

            vis_style = np.vstack(style_images[0].detach().cpu().numpy())
            vis_style = ((vis_style + 1) / 2) * 255

            fakes = self.model.netG.evaluate(style_images, eval_text_encode)
            fake_images = torch.cat(fakes, 1).detach().cpu().numpy()
            real_images = data['img'].detach().cpu().numpy()
            writer_ids = data['wcl'].int().tolist()

            for i, (fake, real, wid, lb, img_id) in enumerate(zip(fake_images, real_images, writer_ids, data['label'], data['idx'])):
                lb = lb.decode()
                ann[f"{wid:03d}"][f'{img_id:05d}'] = lb
                img_id = f'{img_id:05d}.png'

                is_long_tail = any(c in long_tail_chars for c in lb)

                if long_tail_only and not is_long_tail:
                    continue

                fake_img_path = fake_base / f"{wid:03d}" / img_id
                fake_img_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(fake_img_path), 255 * ((fake.squeeze() + 1) / 2))

                if not fake_only:
                    real_img_path = real_base / f"{wid:03d}" / img_id
                    real_img_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(real_img_path), 255 * ((real.squeeze() + 1) / 2))

                counter += 1

            eta = (time.time() - start_time) / (step + 1) * (len(loader) - step - 1)
            eta = str(timedelta(seconds=eta))
            if step % 100 == 0:
                print(f'[{(step + 1) / len(loader) * 100:.02f}%][{counter:05d}] ETA {eta}')

            with open(path / 'ann.json', 'w') as f:
                json.dump(ann, f)

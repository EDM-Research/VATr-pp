import gzip
import json
import os
import pickle
import random
from collections import defaultdict

import PIL
import cv2
import numpy as np
from PIL import Image


TO_MERGE = {
    '.': 'left',
    ',': 'left',
    '!': 'left',
    '?': 'left',
    '(': 'right',
    ')': 'left',
    '\"': 'random',
    "\'": 'random',
    ":": 'left',
    ";": 'left',
    "-": 'random'
}

FILTER_ERR = False


def resize(image, size):
    image_pil = Image.fromarray(image.astype('uint8'), 'L')
    image_pil = image_pil.resize(size)
    return np.array(image_pil)


def get_author_ids(base_folder: str):
    with open(os.path.join(base_folder, "gan.iam.tr_va.gt.filter27"), 'r') as f:
        training_authors = [line.split(",")[0] for line in f]
    training_authors = set(training_authors)

    with open(os.path.join(base_folder, "gan.iam.test.gt.filter27"), 'r') as f:
        test_authors = [line.split(",")[0] for line in f]
    test_authors = set(test_authors)

    assert len(training_authors.intersection(test_authors)) == 0

    return training_authors, test_authors


class IAMImage:
    def __init__(self, image: np.array, label: str, image_id: int, line_id: str, bbox: list = None, iam_image_id: str = None):
        self.image = image
        self.label = label
        self.image_id = image_id
        self.line_id = line_id
        self.iam_image_id = iam_image_id
        self.has_bbox = False
        if bbox is not None:
            self.has_bbox = True
            self.x, self.y, self.w, self.h = bbox

    def merge(self, other: 'IAMImage'):
        global MERGER_COUNT
        assert self.has_bbox, "IAM image has no bounding box information"
        y = min(self.y, other.y)
        h = max(other.y + other.h, self.y + self.h) - y

        x = min(self.x, other.x)
        w = max(self.x + self.w, other.x + other.w) - x

        new_image = np.ones((h, w), dtype=self.image.dtype) * 255

        anchor_x = self.x - x
        anchor_y = self.y - y
        new_image[anchor_y:anchor_y + self.h, anchor_x:anchor_x + self.w] = self.image

        anchor_x = other.x - x
        anchor_y = other.y - y
        new_image[anchor_y:anchor_y + other.h, anchor_x:anchor_x + other.w] = other.image

        if other.x - (self.x + self.w) > 50:
            new_label = self.label + " " + other.label
        else:
            new_label = self.label + other.label
        new_id = self.image_id
        new_bbox = [x, y, w, h]

        new_iam_image_id = self.iam_image_id if len(self.label) > len(other.label) else other.iam_image_id
        return IAMImage(new_image, new_label, new_id, self.line_id, new_bbox, iam_image_id=new_iam_image_id)


def read_iam_lines(base_folder: str) -> dict:
    form_to_author = {}
    with open(os.path.join(base_folder, "forms.txt"), 'r') as f:
        for line in f:
            if not line.startswith("#"):
                form, author, *_ = line.split(" ")
                form_to_author[form] = author

    training_authors, test_authors = get_author_ids(base_folder)

    dataset_dict = {
        'train': defaultdict(list),
        'test': defaultdict(list),
        'other': defaultdict(list)
    }

    image_count = 0

    with open(os.path.join(base_folder, "sentences.txt"), 'r') as f:
        for line in f:
            if not line.startswith("#"):
                line_id, _, ok, *_, label = line.rstrip().split(" ")
                form_id = "-".join(line_id.split("-")[:2])
                author_id = form_to_author[form_id]

                if ok != 'ok' and FILTER_ERR:
                    continue

                line_label = ""
                for word in label.split("|"):
                    if not(len(line_label) == 0 or word in [".", ","]):
                        line_label += " "
                    line_label += word

                image_path = os.path.join(base_folder, "sentences", form_id.split("-")[0], form_id, f"{line_id}.png")

                subset = 'other'
                if author_id in training_authors:
                    subset = 'train'
                elif author_id in test_authors:
                    subset = 'test'

                im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if im is not None and im.size > 1:
                    dataset_dict[subset][author_id].append(IAMImage(
                        im, line_label, image_count, line_id, None
                    ))
                    image_count += 1

    return dataset_dict


def read_iam(base_folder: str) -> dict:
    with open(os.path.join(base_folder, "forms.txt"), 'r') as f:
        forms = [line.rstrip() for line in f if not line.startswith("#")]

    training_authors, test_authors = get_author_ids(base_folder)

    image_info = {}
    with open(os.path.join(base_folder, "words.txt"), 'r') as f:
        for line in f:
            if not line.startswith("#"):
                image_id, ok, threshold, x, y, w, h, tag, *content = line.rstrip().split(" ")
                image_info[image_id] = {
                    'ok': ok == 'ok',
                    'threshold': threshold,
                    'content': " ".join(content) if isinstance(content, list) else content,
                    'bbox': [int(x), int(y), int(w), int(h)]
                }

    dataset_dict = {
        'train': defaultdict(list),
        'test': defaultdict(list),
        'other': defaultdict(list)
    }

    image_count = 0
    err_count = 0

    for form in forms:
        form_id, writer_id, *_ = form.split(" ")
        base_form = form_id.split("-")[0]

        form_path = os.path.join(base_folder, "words", base_form, form_id)

        for image_name in os.listdir(form_path):
            image_id = image_name.split(".")[0]
            info = image_info[image_id]

            subset = 'other'
            if writer_id in training_authors:
                subset = 'train'
            elif writer_id in test_authors:
                subset = 'test'

            if info['ok'] or not FILTER_ERR:
                im = cv2.imread(os.path.join(form_path, image_name), cv2.IMREAD_GRAYSCALE)
                if not info['ok'] and False:
                    cv2.destroyAllWindows()
                    print(info['content'])
                    cv2.imshow("image", im)
                    cv2.waitKey(0)

                if im is not None and im.size > 1:
                    dataset_dict[subset][writer_id].append(IAMImage(
                        im, info['content'], image_count, "-".join(image_id.split("-")[:3]), info['bbox'], iam_image_id=image_id
                    ))
                    image_count += 1
                else:
                    err_count += 1
                    print(f"Could not read image {image_name}, skipping")
            else:
                err_count += 1

    assert not dataset_dict['train'].keys() & dataset_dict['test'].keys(), "Training and Testing set have common authors"

    print(f"Skipped images: {err_count}")

    return dataset_dict


def read_cvl_set(set_folder: str):
    set_images = defaultdict(list)
    words_path = os.path.join(set_folder, "words")

    image_id = 0

    for author_id in os.listdir(words_path):
        author_path = os.path.join(words_path, author_id)

        for image_file in os.listdir(author_path):
            label = image_file.split("-")[-1].split(".")[0]
            line_id = "-".join(image_file.split("-")[:-2])

            stream = open(os.path.join(author_path, image_file), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is not None and image.size > 1:
                set_images[int(author_id)].append(IAMImage(image, label, image_id, line_id))
                image_id += 1

    return set_images


def read_cvl(base_folder: str):
    dataset_dict = {
        'test': read_cvl_set(os.path.join(base_folder, 'testset')),
        'train': read_cvl_set(os.path.join(base_folder, 'trainset'))
    }

    assert not dataset_dict['train'].keys() & dataset_dict[
        'test'].keys(), "Training and Testing set have common authors"

    return dataset_dict

def pad_top(image: np.array, height: int) -> np.array:
    result = np.ones((height, image.shape[1]), dtype=np.uint8) * 255
    result[height - image.shape[0]:, :image.shape[1]] = image

    return result


def scale_per_writer(writer_dict: dict, target_height: int, char_width: int = None) -> dict:
    for author_id in writer_dict.keys():
        max_height = max([image_dict.image.shape[0] for image_dict in writer_dict[author_id]])
        scale_y = target_height / max_height

        for image_dict in writer_dict[author_id]:
            image = image_dict.image
            scale_x = scale_y if char_width is None else len(image_dict.label) * char_width / image_dict.image.shape[1]
            #image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
            image = resize(image, (int(image.shape[1] * scale_x), int(image.shape[0] * scale_y)))
            image_dict.image = pad_top(image, target_height)

    return writer_dict


def scale_images(writer_dict: dict, target_height: int, char_width: int = None) -> dict:
    for author_id in writer_dict.keys():
        for image_dict in writer_dict[author_id]:
            scale_y = target_height / image_dict.image.shape[0]
            scale_x = scale_y if char_width is None else len(image_dict.label) * char_width / image_dict.image.shape[1]
            #image_dict.image = cv2.resize(image_dict.image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
            image_dict.image = resize(image_dict.image, (int(image_dict.image.shape[1] * scale_x), target_height))
    return writer_dict


def scale_word_width(writer_dict: dict):
    for author_id in writer_dict.keys():
        for image_dict in writer_dict[author_id]:
            width = len(image_dict.label) * (image_dict.image.shape[0] / 2.0)
            image_dict.image = resize(image_dict.image, (int(width), image_dict.image.shape[0]))
    return writer_dict


def get_sentences(author_dict: dict):
    collected = defaultdict(list)
    for image in author_dict:
        collected[image.line_id].append(image)

    return [v for k, v in collected.items()]


def merge_author_words(author_words):
    def try_left_merge(index: int):
        if index > 0 and author_words[index - 1].line_id == author_words[index].line_id and not to_remove[index - 1] and not author_words[index - 1].label in TO_MERGE.keys():
            merged = author_words[index - 1].merge(author_words[index])
            author_words[index - 1] = merged
            to_remove[index] = True
            return True
        return False

    def try_right_merge(index: int):
        if index < len(author_words) - 1 and author_words[index].line_id == author_words[index + 1].line_id and not to_remove[index + 1] and not author_words[index + 1].label in TO_MERGE.keys():
            merged = iam_image.merge(author_words[index + 1])
            author_words[index + 1] = merged
            to_remove[index] = True
            return True
        return False

    to_remove = [False for _ in range(len(author_words))]
    for i in range(len(author_words)):
        iam_image = author_words[i]
        if iam_image.label in TO_MERGE.keys():
            merge_type = TO_MERGE[iam_image.label] if TO_MERGE[iam_image.label] != 'random' else random.choice(['left', 'right'])
            if merge_type == 'left':
                if not try_left_merge(i):
                    if not try_right_merge(i):
                        print(f"Could not merge char: {iam_image.label}")
            else:
                if not try_right_merge(i):
                    if not try_left_merge(i):
                        print(f"Could not merge char: {iam_image.label}")

    return [image for image, remove in zip(author_words, to_remove) if not remove], sum(to_remove)


def merge_punctuation(writer_dict: dict) -> dict:
    for author_id in writer_dict.keys():
        author_dict = writer_dict[author_id]

        merged = 1
        while merged > 0:
            author_dict, merged = merge_author_words(author_dict)

        writer_dict[author_id] = author_dict

    return writer_dict


def filter_punctuation(writer_dict: dict) -> dict:
    for author_id in writer_dict.keys():
        author_list = [im for im in writer_dict[author_id] if im.label not in TO_MERGE.keys()]

        writer_dict[author_id] = author_list

    return writer_dict


def filter_by_width(writer_dict: dict, target_height: int = 32, min_width: int = 16, max_width: int = 17) -> dict:
    def is_valid(iam_image: IAMImage) -> bool:
        target_width = (target_height / iam_image.image.shape[0]) * iam_image.image.shape[1]
        if len(iam_image.label) * min_width / 3 <= target_width <= len(iam_image.label) * max_width * 3:
            return True
        else:
            return False

    for author_id in writer_dict.keys():
        author_list = [im for im in writer_dict[author_id] if is_valid(im)]

        writer_dict[author_id] = author_list

    return writer_dict


def write_data(dataset_dict: dict, location: str, height, punct_mode: str = 'none', author_scale: bool = False, uniform_char_width: bool = False):
    assert punct_mode in ['none', 'filter', 'merge']
    result = {}
    for key in dataset_dict.keys():
        result[key] = {}

        subset_dict = dataset_dict[key]

        subset_dict = filter_by_width(subset_dict)

        if punct_mode == 'merge':
            subset_dict = merge_punctuation(subset_dict)
        elif punct_mode == 'filter':
            subset_dict = filter_punctuation(subset_dict)

        char_width = 16 if uniform_char_width else None

        if author_scale:
            subset_dict = scale_per_writer(subset_dict, height, char_width)
        else:
            subset_dict = scale_images(subset_dict, height, char_width)

        for author_id in subset_dict:
            author_images = []
            for image_dict in subset_dict[author_id]:
                author_images.append({
                    'img': PIL.Image.fromarray(image_dict.image),
                    'label': image_dict.label,
                    'image_id': image_dict.image_id,
                    'original_image_id': image_dict.iam_image_id
                })
            result[key][author_id] = author_images

    with open(location, 'wb') as f:
        pickle.dump(result, f)


def write_fid(dataset_dict: dict, location: str):
    data = dataset_dict['test']
    data = scale_images(data, 64, None)
    for author in data.keys():
        author_folder = os.path.join(location, author)
        os.mkdir(author_folder)
        count = 0
        for image in data[author]:
            img = image.image
            cv2.imwrite(os.path.join(author_folder, f"{count}.png"), img.squeeze().astype(np.uint8))
            count += 1


def write_images_per_author(dataset_dict: dict, output_file: str):
    data = dataset_dict["test"]

    result = {}

    for author in data.keys():
        author_images = [image.iam_image_id for image in data[author]]
        result[author] = author_images

    with open(output_file, 'w') as f:
        json.dump(result, f)


def write_words(dataset_dict: dict, output_file):
    data = dataset_dict['train']

    all_words = []

    for author in data.keys():
        all_words.extend([image.label for image in data[author]])

    with open(output_file, 'w') as f:
        for word in all_words:
            f.write(f"{word}\n")


if __name__ == "__main__":
    data_path = r"D:\Datasets\IAM"
    fid_location = r"E:/projects/evaluation/shtg_interface/data/reference_imgs/h64/iam"
    height = 32
    data_collection = {}

    output_location = r"E:\projects\evaluation\shtg_interface\data\datasets"

    data = read_iam(data_path)
    test_data = dict(scale_word_width(data['test']))
    train_data = dict(scale_word_width(data['train']))
    test_data.update(train_data)
    for key, value in test_data.items():
        for image_object in value:
            if len(image_object.label) <= 0 or image_object.image.size == 0:
                continue
            data_collection[image_object.iam_image_id] = {
                'img': image_object.image,
                'lbl': image_object.label,
                'author_id': key
            }

    with gzip.open(os.path.join(output_location, f"iam_w16_words_data.pkl.gz"), 'wb') as f:
        pickle.dump(data_collection, f)

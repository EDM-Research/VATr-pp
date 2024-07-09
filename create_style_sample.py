import os
import argparse

import cv2
from util.vision import get_page, get_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True, default='files/style_samples/00')

    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    image = cv2.resize(image, (image.shape[1], image.shape[0]))
    result = get_page(image)
    words, _ = get_words(result)

    output_path = args.output_folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i, word in enumerate(words):
        cv2.imwrite(os.path.join(output_path, f"word{i}.png"), word)

import os

import cv2
from util.vision import get_page, get_words


if __name__ == "__main__":
    path = "files/page.JPEG"
    style_name = "bram2"

    image = cv2.imread(path)
    image = cv2.resize(image, (image.shape[1], image.shape[0]))
    result = get_page(image)
    words, _ = get_words(result)

    output_path = os.path.join("files/style_samples", style_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i, word in enumerate(words):
        cv2.imwrite(os.path.join(output_path, f"word{i}.png"), word)

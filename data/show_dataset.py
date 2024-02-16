import os
import pickle
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import get_transform


def summarize_dataset(data: dict):
    print(f"Training authors: {len(data['train'].keys())} \t Testing authors: {len(data['test'].keys())}")
    training_images = sum([len(data['train'][k]) for k in data['train'].keys()])
    testing_images = sum([len(data['test'][k]) for k in data['test'].keys()])
    print(f"Training images: {training_images} \t Testing images: {testing_images}")


def compare_data(path_a: str, path_b: str):
    with open(path_a, 'rb') as f:
        data_a = pickle.load(f)
        summarize_dataset(data_a)

    with open(path_b, 'rb') as f:
        data_b = pickle.load(f)
        summarize_dataset(data_b)

    training_a = data_a['train']
    training_b = data_b['train']

    training_a = {int(k): v for k, v in training_a.items()}
    training_b = {int(k): v for k, v in training_b.items()}

    while True:
        author = random.choice(list(training_a.keys()))

        if author in training_b.keys():
            author_images_a = [np.array(im_dict["img"]) for im_dict in training_a[author]]
            author_images_b = [np.array(im_dict["img"]) for im_dict in training_b[author]]

            labels_a = [str(im_dict["label"]) for im_dict in training_a[author]]
            labels_b = [str(im_dict["label"]) for im_dict in training_b[author]]

            vis_a = np.hstack(author_images_a[:10])
            vis_b = np.hstack(author_images_b[:10])

            cv2.imshow("Author a", vis_a)
            cv2.imshow("Author b", vis_b)

            cv2.waitKey(0)

        else:
            print(f"Author: {author} not found in second dataset")


def show_dataset(path: str, samples: int = 10):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        summarize_dataset(data)

    training = data['train']

    author = training['013']
    author_images = [np.array(im_dict["img"]).astype(np.uint8) for im_dict in author]

    for img in author_images:
        cv2.imshow('image', img)
        cv2.waitKey(0)

    for author in list(training.keys()):

        author_images = [np.array(im_dict["img"]).astype(np.uint8) for im_dict in training[author]]
        labels = [str(im_dict["label"]) for im_dict in training[author]]

        vis = np.hstack(author_images[:samples])
        print(f"Author: {author}")
        cv2.destroyAllWindows()
        cv2.imshow("vis", vis)
        cv2.waitKey(0)


def test_transform(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        summarize_dataset(data)

    training = data['train']
    transform = get_transform(grayscale=True)

    for author_id in training.keys():
        author = training[author_id]
        for image_dict in author:
            original_image = image_dict['img'].convert('L')
            transformed_image = transform(original_image).detach().numpy()
            restored_image = (((transformed_image + 1) / 2) * 255).astype(np.uint8)
            restored_image = np.squeeze(restored_image)
            original_image = np.array(original_image)

            wrong_pixels = (original_image != restored_image).astype(np.uint8) * 255

            combined = np.hstack((restored_image, original_image, wrong_pixels))

            cv2.imshow("original", original_image)
            cv2.imshow("restored", restored_image)
            cv2.imshow("combined", combined)

            f, ax = plt.subplots(1, 2)
            ax[0].hist(original_image.flatten())
            ax[1].hist(restored_image.flatten())
            plt.show()

            cv2.waitKey(0)

def dump_words():
    data_path = r"..\files\IAM-32.pickle"

    p_mark = 'point'
    p = '.'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    training = data['train']

    target_folder = f"../saved_images/debug/{p_mark}"

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    os.mkdir(target_folder)

    count = 0

    for author in list(training.keys()):

        author_images = [np.array(im_dict["img"]).astype(np.uint8) for im_dict in training[author]]
        labels = [str(im_dict["label"]) for im_dict in training[author]]

        for img, label in zip(author_images, labels):
            if p in label:
                cv2.imwrite(os.path.join(target_folder, f"{count}.png"), img)
                count += 1


if __name__ == "__main__":
    test_transform("../files/IAM-32.pickle")
    #show_dataset("../files/IAM-32.pickle")
    #compare_data(r"../files/IAM-32.pickle", r"../files/_IAM-32.pickle")

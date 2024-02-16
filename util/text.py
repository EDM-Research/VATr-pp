import math
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import string
from abc import ABC, abstractmethod
from functools import partial


class TextGenerator(ABC):
    def __init__(self, max_lenght: int = None):
        self.max_length = max_lenght

    @abstractmethod
    def generate(self):
        pass


class AugmentedGenerator(TextGenerator):
    def __init__(self, strength: float, alphabet: list, max_lenght: int = None):
        super().__init__(max_lenght)
        self.strength = strength
        self.alphabet = list(alphabet)
        if "%" in alphabet:
            self.alphabet.remove("%")

    @abstractmethod
    def generate(self):
        pass

    def set_strength(self, strength: float):
        self.strength = strength

    def get_strength(self):
        return self.strength


class ProportionalAugmentedGenerator(AugmentedGenerator):
    def __init__(self, max_length: int, generator: TextGenerator, alphabet: list, strength: float = 0.5):
        super().__init__(strength, alphabet, max_length)
        self.generator = generator

        self.char_stats = {}
        self.sampling_probs = {}
        self.init_statistics()

    def init_statistics(self):
        char_occurrences = {k: 0 for k in self.alphabet}
        character_count = 0

        for _ in range(10000):
            word = self.generator.generate()
            for char in word:
                char_occurrences[char] += 1
                character_count += 1

        self.char_stats = {k: v / character_count for k, v in char_occurrences.items()}
        scale = max([v for v in self.char_stats.values()])
        self.char_stats = {k: v / scale for k, v in self.char_stats.items()}
        self.sampling_probs = {k: 1.0 - v for k, v in self.char_stats.items()}

    def random_char(self):
        return random.choices(list(self.sampling_probs.keys()), weights=list(self.sampling_probs.values()), k=1)[0]

    def generate(self):
        word = self.generator.generate()
        word = self.augment(word)
        return word

    def augment(self, word):
        probs = np.random.rand(len(word))
        target_probs = [self.strength * self.char_stats[c] for c in word]

        replace = probs < target_probs

        for index in range(len(word)):
            if replace[index]:
                char = self.random_char()
                word = set_char(word, char, index)
        return word


class FileTextGenerator(TextGenerator):
    def __init__(self, max_length: int, file_path: str, alphabet: list):
        super().__init__(max_length)

        with open(file_path, 'r') as f:
            self.words = f.read().splitlines()
        self.words = [l for l in self.words if len(l) < self.max_length and set(l) <= set(alphabet)]

    def generate(self):
        return random.choice(self.words)


class CVLFileTextIterator(TextGenerator):
    def __init__(self, max_length: int, file_path: str, alphabet: list):
        super().__init__(max_length)

        self.words = []

        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                _, *annotation = line.rstrip().split(",")
                annotation = ",".join(annotation)
                self.words.append(annotation)
        self.words = [l for l in self.words if len(l) < self.max_length and set(l) <= set(alphabet)]
        self.index = 0

    def generate(self):
        word = self.words[self.index % len(self.words)]
        self.index += 1
        return word


def set_char(s, character, location):
    return s[:location] + character + s[location + 1:]


class GibberishGenerator(TextGenerator):
    def __init__(self, max_length: int = None):
        super().__init__(max_length)
        self.lower_case = list(string.ascii_lowercase)
        self.upper_case = list(string.ascii_uppercase)
        self.special = list(' .-\',"&();#:!?+*/')
        self.numbers = [str(i) for i in range(10)]

    def get_word_length(self) -> int:
        length = int(math.ceil(np.random.chisquare(8)))
        while self.max_length is not None and length > self.max_length:
            length = int(math.ceil(np.random.chisquare(8)))
        return length

    def generate(self):
        return self.generate_random()

    def generate_random(self):
        alphabet = self.upper_case + self.lower_case + self.special + self.numbers
        string = ''.join(random.choices(alphabet, k=self.get_word_length()))

        return string


class IAMTextGenerator(TextGenerator):
    def generate(self):
        return random.choice(self.words)

    def __init__(self, max_length: int, path: str, subset: str = 'train'):
        super().__init__(max_length)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        data = data[subset]
        self.words = []
        for author_id in data.keys():
            for image_dict in data[author_id]:
                if len(image_dict['label']) <= self.max_length:
                    self.words.append(image_dict['label'])


def get_generator(args):
    if args.corpus == "standard":
        if args.english_words_path.endswith(".csv"):
            generator = CVLFileTextIterator(20, args.english_words_path, args.alphabet)
        else:
            generator = FileTextGenerator(20, args.english_words_path, args.alphabet)
    else:
        generator = IAMTextGenerator(20, "files/IAM-32.pickle", 'train')

    if args.text_augment_strength > 0:
        if args.text_aug_type == 'proportional':
            return ProportionalAugmentedGenerator(20, generator, args.alphabet, args.text_augment_strength)
        elif args.text_aug_type == 'gibberish':
            return GibberishGenerator(20)
        else:
            return ProportionalAugmentedGenerator(20, generator, args.alphabet, args.text_augment_strength)

    return generator


if __name__ == "__main__":
    alphabet = list('Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%')
    original_generator = FileTextGenerator(max_length=20, file_path="../files/english_words.txt", alphabet=alphabet)
    gib = ProportionalAugmentedGenerator(20, original_generator, alphabet=alphabet, strength=0.5)

    generated_words = []

    for _ in range(1000):
        word = gib.generate()
        generated_words.append(len(word))
        if len(set(word)) < len(word):
            print(word)

    plt.hist(generated_words)
    plt.show()
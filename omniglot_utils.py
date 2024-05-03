import os

import matplotlib.pyplot as plt

background_imgs_path = 'omniglot/images_background'
eval_imgs_path = 'omniglot/images_evaluation'


def get_chars_from_dir(dir_path):
    chars = []
    for alphabet in os.listdir(dir_path):
        path = os.path.join(dir_path, alphabet)
        if os.path.isdir(path):
            for char in os.listdir(path):
                sub_path = os.path.join(path, char)
                if os.path.isdir(sub_path):
                    chars.append(sub_path)

    return chars


def get_all_chars():
    all_chars = []
    all_chars += get_chars_from_dir(background_imgs_path)
    all_chars += get_chars_from_dir(eval_imgs_path)

    return all_chars


def test_sync(x, y):
    for img, (idx, cls) in zip(x, enumerate(y)):
        img = img.permute(1, 2, 0)
        img = img.numpy()
        plt.imshow(img)
        plt.title(str(cls.item()) + str(idx % 5))
        plt.show()


def viz_logit(x, y, logits):
    for idx, (img, cls, logit) in enumerate(zip(x, y, logits)):
        img = img.permute(1, 2, 0)
        img = img.numpy()
        plt.imshow(img)
        title = "logit: " + str(logit.tolist()) \
                + "\nclass: " + str(cls)
        plt.title(title)
        plt.show()

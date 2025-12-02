# -*-coding:utf8-*-

import os
import argparse
from PIL import Image
import pickle


def check_sample(img_path):
    if not os.path.exists(img_path):
        return 'not-exists'
    img = Image.open(img_path)
    if img.mode != 'RGB':
        return 'not-rgb'
    return None


def main(opts):
    cwd = os.getcwd()
    train_split = os.path.join(cwd, 'train_100.txt')
    with open(train_split, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    train_images = []
    for line in lines:
        img_path = line.split(' ')[0]
        train_images.append(os.path.join(opts.data_path, img_path))
    bad_file = os.path.join(cwd, 'bad_train_image.txt')
    bad_train_cnt = 0
    with open(bad_file, 'wb') as fw:
        for fi in train_images:
            r = check_sample(fi)
            if r is not None:
                pickle.dump([fi, r], fw)
                bad_train_cnt += 1
    print('bad train images:', bad_train_cnt)
    val_split = os.path.join(cwd, 'val_100.txt')
    with open(val_split, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    val_images = []
    for line in lines:
        img_path = line.split(' ')[0]
        val_images.append(os.path.join(opts.data_path, img_path))
    bad_file = os.path.join(cwd, 'bad_val_image.txt')
    bad_val_cnt = 0
    with open(bad_file, 'wb') as fw:
        for fi in val_images:
            r = check_sample(fi)
            if r is not None:
                pickle.dump([fi, r], fw)
                bad_val_cnt += 1
    print('bad validation images:', bad_val_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Check data')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    main(opts=args)

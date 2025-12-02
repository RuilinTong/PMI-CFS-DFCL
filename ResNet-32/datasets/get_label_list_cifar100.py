# -*-coding:utf8-*-

import argparse
import os
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get cifar100 class list')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--class_order_file', type=str)
    args = parser.parse_args()
    cwd = os.getcwd()

    dataset = torchvision.datasets.CIFAR100(root=args.data_path, download=True, transform=None, train=True)
    with open(args.class_order_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    class_order = [int(ci) for ci in lines[0].split(', ')]
    # print(dataset.class_to_idx)
    idx2class = {}
    for cls in dataset.class_to_idx.keys():
        idx2class[dataset.class_to_idx[cls]] = cls
    classes = []
    for idx, i in enumerate(class_order):
        classes.append([idx, i, idx2class[i]])
    out_file = os.path.join(cwd, 'classes_cifar100.txt')
    with open(out_file, 'w', encoding='utf8') as fw:
        for ci in classes:
            fw.write(str(ci[0]))
            fw.write('\t\t')
            fw.write(str(ci[1]))
            fw.write('\t\t')
            fw.write(ci[2])
            fw.write('\n')


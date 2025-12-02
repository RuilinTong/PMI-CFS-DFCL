# -*-coding:utf8-*-

import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get tinyimagenet label')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--class_order_file', type=str)
    args = parser.parse_args()
    cwd = os.getcwd()

    with open(args.class_order_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    class_order = [int(ci) for ci in lines[0].split(', ')]
    wnid_file = os.path.join(args.data_path, 'wnids.txt')
    with open(wnid_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    wnids = {v: k for v, k in enumerate(lines)}  # label2wnid
    wnid_set = set()
    for line in lines:
        wnid_set.add(line)
    words_file = os.path.join(args.data_path, 'words.txt')
    with open(words_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    wnid2class = {}
    for line in lines:
        wnid, cls = line.split('\t')
        if wnid in wnid_set:
            wnid2class[wnid] = cls
    out_file = os.path.join(cwd, 'classes_tinyimagenet.txt')
    with open(out_file, 'w', encoding='utf8') as fw:
        for i, ci in enumerate(class_order):
            cls = wnid2class[wnids[ci]]
            fw.write(str(i))
            fw.write('\t\t')
            fw.write(str(ci))
            fw.write('\t\t')
            fw.write(cls)
            fw.write('\n')

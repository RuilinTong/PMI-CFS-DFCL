# -*-coding:utf8-*-

import os
import sys


if __name__ == '__main__':
    target_path = sys.argv[1]
    for fi in os.listdir(target_path):
        if 'test' in fi:
            for gi in os.listdir(os.path.join(target_path, fi)):
                if '_' in gi and '.png' in gi:
                    os.remove(os.path.join(target_path, fi, gi))
                if '.pkl' in gi:
                    os.remove(os.path.join(target_path, fi, gi))

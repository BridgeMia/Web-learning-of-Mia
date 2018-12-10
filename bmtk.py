"""
Some useful functions

Some of the functions are borrowed from LJQ, please refer to
https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/ljqpy.py

"""

import re
import numpy as np
import os

def write_line(fout, lst):
    fout.write('\t'.join([str(x) for x in lst]) + '\n')

def is_chs_str(z):
    return re.search('^[\u4e00-\u9fa5]+$', z) is not None

def loadCSV(fn):
    ret = []
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            ret.append(lln)
    return ret

def saveCSV(csv, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            write_line(fout, x)


def load_list(fn):
    with open(fn, encoding="utf-8") as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st


def load_dict(fn, func=str):
    dict = {}
    with open(fn, encoding="utf-8") as fin:
        for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
            dict[lv[0]] = func(lv[1])
    return dict


def save_dict(dict, ofn, output0=True):
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in dict.keys():
            if output0 or dict[k] != 0:
                fout.write(str(k) + "\t" + str(dict[k]) + "\n")


def save_list(st, ofn):
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in st:
            fout.write(str(k) + "\n")

# Longest Child Substring
def lcs(s1, s2):
    matrix = np.zeros(shape = [len(s1) + 1, len(s2) + 1])
    cursor = 0
    mmax = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                matrix[i+1][j+1] = matrix[i][j]+1
                if matrix[i+1][j+1] > mmax:
                    mmax = matrix[i+1][j+1]
                    cursor = i+1
    mmax = int(mmax)
    return s1[cursor-mmax:cursor], mmax

# Ergodic a dir
class ErgodicDir:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.files = []

    def ergodic(self,dir_name):
        for item in os.listdir(dir_name):
            path = os.path.join(dir_name, item)
            if os.path.isdir(path):
                self.ergodic(path)
            else:
                self.files.append(path)

    def run(self):
        self.ergodic(self.root_dir)

    def export(self):
        return self.files

def list_split_g(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    ergodic_dir_sp = ErgodicDir(r'sourse/testdir')
    ergodic_dir_sp.run()
    x = ergodic_dir_sp.export()
    print(x)


#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

"""
Synthetic recognition datasets with Tsinghua-CTW dataset.
"""
import os
import json

from PIL import Image


# TODO:  split ctw chars; gen new data.
# Done: Gen CTW alphabet; calculate alphabet union.

def gen_alphabet(jsonl_path, alphabet_out_path="ctw_alphabet.txt", encoding="GBK"):
    alphabet = []
    attr = []
    with open(jsonl_path, encoding=encoding) as f:
        for line in f:
            json_line = json.loads(line)
            img_fn = json_line["file_name"]
            annos = json_line["annotations"]
            for annol in annos:
                for anno in annol:
                    alphabet += anno["text"]
                    for a in anno["attributes"]:
                        attr.append(a)
    return list(set(alphabet)), list(set(attr))


def write_alphabet(alphabet, filename, encoding="GBK"):
    with open(filename, "w", encoding=encoding) as f:
        for line in alphabet:
            f.write(line + '\n')


def load_alphabet_txt(filename, encoding="GB18030"):
    alphabet = []
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            alphabet.append(line.strip())
    return alphabet


def crop_character(jsonl_path, exclude_alphabet, image_path="./images_trainval/",
                   out_path="./ctw-char/", encoding="GB18030"):
    stat = 0
    with open(jsonl_path, encoding=encoding) as f:
        for line in f:
            json_line = json.loads(line)
            img_fn = json_line["file_name"]
            print("Cropping from ", img_fn)
            annos = json_line["annotations"]
            im = Image.open(image_path + img_fn)
            for annol in annos:
                for anno in annol:
                    char = anno["text"]
                    if char not in exclude_alphabet:
                        attr_str = ""
                        for att in anno["attributes"]:
                            attr_str += att[0]
                        area = anno["adjusted_bbox"]  # AABB Convention: [x, y, w, h]
                        char_im = im.crop((int(area[0]), int(area[1]), int(area[2] + area[0]), int(area[3] + area[1])))
                        try:
                            os.mkdir(out_path + char)
                        except FileExistsError:
                            pass
                        char_im.save(out_path + char + '/' + str(stat) + '_' + attr_str + '_' + img_fn)
                        stat += 1
                    # else:
                    #     print("Skipped char ", char)
            print("Cropping fin, stat ", stat)


def synthetic_sample_gen(alphabet, sentences, out_dir="./STV2KSynOnCTW/"):
    """
    TODO: this.
    Preprocess sentences, select chars, preprocess and augment chars, concatenate chars, save imgs and gts.
    """
    pass


if __name__ == "__main__":
    exc_alpha = load_alphabet_txt("ctw_not_in_stv2k_alphabet_1221-GB18030.txt")
    crop_character("./ctw-annotations/train.jsonl", exc_alpha)

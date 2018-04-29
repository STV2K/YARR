#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

"""
Synthetic recognition datasets with Tsinghua-CTW dataset.
"""
import os
import json

import numpy as np
from PIL import Image

import config
import data_util as du

# TODO:  split ctw chars; gen new data.
# Done: Gen CTW alphabet; calculate alphabet union.

def gen_ctw_alphabet(jsonl_path, alphabet_out_path="ctw_alphabet.txt", encoding="GBK"):
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
    with open("Err_rec", "w", encoding=encoding) as fl:
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
                            try:
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
                            except:
                                fl.write(str(img_fn) + "\t" + str(anno) + "\n")
                                continue
                        # else:
                        #     print("Skipped char ", char)
                print("Cropping fin, stat ", stat)


# def get_image_list(file_path):
#     file_list = []
#     for ext in ['jpg', 'png', 'jpeg', 'JPG']:
#         for root, dirs, files in os.walk(file_path):
#             for f in files:
#                 # Only check files under root
#                 if f.split(".")[-1] in ext and root == file_path:
#                     file_list.append(os.path.join(file_path, f))
#     return file_list


def synthetic_sample_gen(alphabet, sentences, gen_height_min=16, gen_height_max=28, re_choose_prob=0.90,
                         width_scale=(0.7, 1.6), better_choose_prob=0.5, encoding="GBK",
                         out_dir="./stvs_on_ctwc/", src_dir="./ctw-char/", fn_prefix="stvctw_"):
    """
    TODO: augment and normalize chars; adopt better policy to calculate target size for char.
    Preprocess sentences, select chars, preprocess and augment chars, concatenate chars, save imgs and gts.
    """
    index = 0
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    for sentence in sentences:
        filtered = ""
        now_height = 0
        sentence_array = None
        out = True
        for char in sentence:
            if char in alphabet:
                if char != '/' and char != '.':
                    filtered += char
        print("Generating ", filtered)
        for char in filtered:
            try:
                # if char == '/':
                #     char = 'slash'
                # elif char == '.':
                #     char = 'dot'
                char_list = du.get_image_list(src_dir + char + '/')
                # chances to filter out some low resolution char images
                assert len(char_list) > 0
                while len(char_list) > 0:
                    random_im = np.random.choice(char_list)
                    char_list.remove(random_im)
                    char_im = Image.open(random_im)
                    w, h = char_im.size
                    if (w < gen_height_min // 2 or h < gen_height_min ) \
                            and np.random.random() < re_choose_prob:
                        continue
                    else:
                        break
                if now_height == 0:
                    now_height = h
                    if now_height < gen_height_min:
                        now_height = gen_height_min
                    elif now_height > gen_height_max:
                        now_height = gen_height_max
                    sentence_array = np.zeros((now_height, 0, 3)).astype(np.uint8)
                if w < now_height:
                    w_scale = np.random.random() * (width_scale[1] - 1) + 1
                else:
                    w_scale = np.random.random() * (1 - width_scale[0]) + width_scale[0]
                w_resize = int(w_scale * w) + 1
                # print((w_resize, now_height))
                # Change resampler to BILINEAR if too slow
                char_im = char_im.resize((w_resize, now_height), Image.ANTIALIAS)
                # print("Im size ", char_im.size)
                char_array = np.array(char_im).astype(np.uint8)  # Convention for array: (h, w, c)
                # print("Arr sizes ", sentence_array.shape, char_array.shape)
                sentence_array = np.concatenate((sentence_array, char_array), 1)
            except AssertionError:
                print("No image for char:", char, "in sentence ", sentence)
                out = False
            except Exception as _E:
                print("Err on char:", char, "in sentence ", sentence, "\n", _E)
                out = False
        if out and len(filtered) > 0:
            sentence_im = Image.fromarray(sentence_array)
            sentence_im.save(out_dir + fn_prefix + str(index) + ".jpg")
            with open(out_dir + fn_prefix + str(index) + ".txt", "w", encoding=encoding) as gtf:
                gtf.write(filtered)
            index += 1


def sentences_stv_gen(paths):
    alphabet = []
    for path in paths:
        img_list = du.get_image_list(path)

        for img_path in img_list:
            img_filename = str(os.path.basename(img_path))
            label_path = img_path.replace(img_filename.split('.')[1], 'txt')
            _, content, __ = du.load_annotation(label_path)
            for line in content:
                if line != '':
                    alphabet.append(line.strip())
    return alphabet


if __name__ == "__main__":
    # exc_alpha = load_alphabet_txt("ctw_not_in_stv2k_alphabet_1221-GB18030.txt")
    sentences_tr = sentences_stv_gen([config.training_data_path_pami2])
    # sentences_ts = sentences_stv_gen([config.test_data_path_pami2])
    int_alpha = load_alphabet_txt("stv-ctw-intersection-2547-GBK.txt", "GBK")
    # crop_character("./ctw-annotations/train.jsonl", exc_alpha)
    synthetic_sample_gen(int_alpha, sentences_tr)


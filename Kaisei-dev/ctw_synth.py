#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

"""
Synthetic recognition datasets with Tsinghua-CTW dataset.
"""
import json


# TODO:  split ctw chars; gen new data.
# Done: Gen CTW alphabet; calculate alphabet union.

def gen_alphabet(jsonl_path, alphabet_path="ctw_alphabet.txt", encoding="GBK"):
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



import glob
import os
import numpy as np
import tensorflow as tf
import config_utils as config

def get_images():
    files = []
    for root, dirs, filenames in os.walk(config.FLAGS.training_data_path):
        for file in filenames:
            if not file.endswith('.jpg'):
                continue
            files.append(os.path.join(root, file))
    print('{} training images in {}'.format(len(files), config.FLAGS.training_data_path))
    return files

def load_annotation(ano_path):
    polys = []
    tags = []
    objIndex = 0
    datas = ""
    gt_word = ""
    with open(ano_path, 'r', encoding="gbk") as file:
        for line in file.readlines():
            objIndex += 1

            line_num = objIndex % 3
            if line_num == 1:
                data = line
                continue
            elif line_num == 2:
                gt_word = line.strip('\r\n')
            else:
                continue

            datas = data.split(',')
            nums = []
            for num in datas:
                nums.append(int(num))
            x1, y1, x2, y2, x3, y3, x4, y4 = nums
            polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

            if gt_word != "" and gt_word != "\n" and gt_word != "\r\n":
                tags.append(True)
            else:
                tags.append(False)

    return np.array(polys, dtype=np.float32), np.array(tags, dtype=np.bool)

if __name__ == '__main__':
    # get_images()
    polys = load_annotation(config.FLAGS.training_data_path + 'STV2K_tr_0001.txt')
    print(polys)

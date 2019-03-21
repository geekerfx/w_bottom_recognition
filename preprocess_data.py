import pandas as pd
import numpy as np
import os
from skimage import io, transform
import warnings
import json


def expand_data(line):
    """
    通过添加随机噪声对数据进行扩充处理
    第一中是在纵向上，给每个点添加一定的随机噪声
    第二种是在横向上，在每两个点之间添加（mean + noise)点
    :param line: 行数据
    :return: list1, list2
    """
    result1 = [(x + np.random.random(1) * 0.1) for x in line]
    result2 = []
    for idx in range(1, len(line)):
        tmp = (line[idx] + line[idx - 1]) / 2
        tmp += np.random.uniform(-1, 1, [1]) * tmp * 0.1
        result2.append(tmp)
        result2.append(line[idx])
    return result1, result2


def resize(root_path, img_name, out_path, osize):
    """
    对指定文件夹中的所有图片进行尺寸调整
    :param src_path:  未经处理的图片的根目录
    :param out_path:  处理过的图片的输出目录
    :param osize: 输出的大小
    :return: None
    """
    # change channels from RGBA to RGB
    img = io.imread(os.path.join(root_path, img_name))[:, :, :3]
    img = transform.resize(osize)
    io.imsave(os.path.join(out_path, img_name), img)


def generate_tags(tag_src_path, ufilter=None):
    img_names, labels = [], []
    for img_name in os.listdir(tag_src_path):
        label = int(img_name.split("_")[0])
        img_names.append(img_name)
        labels.append(label)
    df = pd.DataFrame({"img_name": img_names, "label": labels})
    return df


def main():
    conf2 = json.load(open('conf/paint_preprocess.json', 'r', encoding='UTF-8'))['preprocess']
    # src_path = conf2['src_path']
    # out_path = conf2['out_path']
    # size = conf2['size']
    # for img_name in os.listdir(src_path):
    #     resize(src_path, img_name, out_path, size)

    tag_out_path = conf2['tag_out_path']
    tag_src_path = conf2['tag_src_path']
    df = generate_tags(tag_src_path)
    os.makedirs(tag_out_path, exist_ok=True)
    df.to_csv(os.path.join(tag_out_path, "labels.csv"), index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

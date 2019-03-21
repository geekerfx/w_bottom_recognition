from torch.utils.data import Dataset
import os
from PIL import Image


class DataSetW(Dataset):
    def __init__(self, df, img_path, transformer=None):
        """
        初始化一个W底数据集对象
        :param df: 包含图片名称以及对应的类别的DataFrame对象
        :param img_path: 图片的目录
        :param transformer: 图片的转换器
        """
        self.df = df
        self.img_path = img_path
        self.transformer = transformer

    def __getitem__(self, idx):
        img_name, img_class = self.df['img_name'][idx], self.df['label'][idx]
        img = Image.open(os.path.join(self.img_path, img_name))
        if self.transformer:
            return self.transformer(img), img_class
        return img, img_class

    def __len__(self):
        return len(self.df)



import torch
import torch.utils.data as data
from imageio import imread
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from os import listdir
from os.path import splitext
import random
import json
import cv2
import pandas as pd

TRAIN_PATH = '../data_list/train_df.csv'
TEST_PATH = '../data_list/test_df.csv'

class BasicDataset(data.Dataset):
    def __init__(self, data_path, data_type='train', transform=None):
        self.data_path = Path(data_path)
        self.data_type = data_type

        assert data_type == 'train' or data_type == 'test', f"You have to input data type \'train\' or \'test\'"

        if data_type == 'train':
            self.data_info = pd.read_csv(TRAIN_PATH)
        elif data_type == 'test':
            self.data_info = pd.read_csv(TEST_PATH)

        self.ids = [splitext(file)[0] for file in self.data_info.loc[:,'ImageId']]

        if not self.ids:
            raise RuntimeError(f'No input file found in {data_path}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @classmethod
    def rle_to_pixels(cls, rle_code):
        '''
        Transforms a RLE code string into a list of pixels of a (768, 768) canvas
        '''
        # Divide the rle in a list of pairs of ints rapresenring the (start,lenght)
        rle_code = [int(i) for i in rle_code.split()]

        pixels = [
            # Find the 2d coordinate for the canva using the mod function (%) and the integer division function(//)
            (pixel_position % 768, pixel_position // 768)
            # I select the start pixel and the lenght of the line
            for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
            # I screen all the pixel positions rapresenting (start,end)
            for pixel_position in range(start, start + length)]
        return pixels

    def load(self, filename, img_name):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.jpg']:
            # array 형태로 이미지 로드
            load_img = np.array(Image.open(filename))

            # 검정색 색공간 생성.
            canvas = np.zeros((load_img.shape[0], load_img.shape[1]), np.int32) # 어차피 나중에 true mask를 float형태로 변환해주니 이 부분은 굳이 안바꿔도 될듯하다.
            pixels = self.rle_to_pixels(self.data_info[self.data_info['ImageId'] == img_name]['EncodedPixels']) # 해당 이미지의 masking 정보를 가져온다.
            canvas[tuple(zip(*pixels))] = 1 # 마스킹 해줌 (row,col)형태의 array

            return Image.open(filename), Image.fromarray(canvas)

    def __getitem__(self, index):
        # 데이터를 불러온 다음 transform 시켜서 반환할 수 있도록 하자.
        name = self.ids[index]
        img_file = list(self.data_path.glob(name + '.jpg'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # 이미지 데이터 로딩
        img, mask = self.load(img_file[0],name)

        # 전처리
        sample = {'image': img, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return sample

    def __len__(self):
        return len(self.ids)

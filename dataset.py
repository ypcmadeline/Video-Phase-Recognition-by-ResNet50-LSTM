"""Dataset.

    Customize your dataset here.
"""

import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# from torchvision import transforms
import glob
import re


class Surgery(Dataset):
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.data = []

        if train == True:
        
            files = glob.glob("datas/annotation" + "/*.csv")
            for f in files:
                if f == "datas/annotation\\video_41.csv":
                    continue
                df = pd.read_csv(f)
                d = re.findall(r'\d+', f)
                df["video"] = d[0]
                self.data.append(df)
                print(f)
            
            self.data = pd.concat(self.data)

            print(sum(self.data.Phase==0))
            print(sum(self.data.Phase==1))
            print(sum(self.data.Phase==2))
            print(sum(self.data.Phase==3))
            print(sum(self.data.Phase==4))
            print(sum(self.data.Phase==5))
            print(sum(self.data.Phase==6))

            rows = self.data.values.tolist()
            # print(len(rows))
            oversampled_rows = [row for row in rows for _ in range(get_sample_ratio(row))]
            # print(len(oversampled_rows))
            self.data = pd.DataFrame(oversampled_rows, columns=self.data.columns)

        else:
            self.data = pd.read_csv("datas/annotation/video_41.csv")




       

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        data = self.data.iloc[index]
        if self.train == True:
            video = data[2]
        else:
            video = 41
        image = Image.open(f"datas/images/{video}/{data[0]}")
        if self.transform:
            image = self.transform(image)
        label = data[1]
        return image, label

    def __len__(self):
        return len(self.data)


def get_sample_ratio(row):
    ratio = [2,1,2,1,5,2,2]
    return ratio[row[1]]

if __name__ == '__main__':
    df = pd.read_csv(f"datas/annotation/video_1.csv")
    # ratio = [1,1,3,4,1,15,11]
    # rows = df_train.values.tolist()
    # print(len(rows))
    # oversampled_rows = [row for row in rows for _ in range(get_sample_ratio(row))]
    # print(len(oversampled_rows))
    # df_train_oversampled = pd.DataFrame(oversampled_rows, columns=df_train.columns)
    # print(df_train_oversampled)
    # # print(df_train_oversampled)
    # files = glob.glob("datas/annotation" + "/*.csv")
    # for f in files:
    #     df = pd.read_csv(f)
    #     d = re.findall(r'\d+', f)
    #     df["video"] = d[0]
    #     print(df)
    s = Surgery()
    

    


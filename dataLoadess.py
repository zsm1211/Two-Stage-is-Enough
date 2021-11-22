from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import numpy as np
import random

class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            dir_list = os.listdir(path)
            groung_truth_path = path
            measurement_path = path + '/measurement'

            if os.path.exists(groung_truth_path):
                groung_truth = os.listdir(groung_truth_path)
                # measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):
        # print(index)
        groung_truth= self.data[index]["groung_truth"]

        gt = scio.loadmat(groung_truth)
        gt=torch.from_numpy(gt['patch_save']/255)
        # meas = scio.loadmat(measurement)
        # if "patch_save" in gt:
        #     gt = torch.from_numpy(gt['patch_save'] / 255)
        # elif "p1" in gt:
        #     gt = torch.from_numpy(gt['p1'] / 255)
        # elif "p2" in gt:
        #     gt = torch.from_numpy(gt['p2'] / 255)
        # elif "p3" in gt:
        #     gt = torch.from_numpy(gt['p3'] / 255)

        # meas = torch.from_numpy(meas['meas'] / 255)

        gt = gt.permute(2, 0, 1)

        # print(tran(img).shape)

        return gt

    def __len__(self):

        return len(self.data)

def iter_files(rootDir):
    img_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            img_list.append(file_name)
            #print(file_name)
    return img_list

def load_data(img_list):
    data_list = []
    #for i in range(len(img_list)):
    for i in range(len(img_list)):
    # for i in range(10):
        img = scio.loadmat(img_list[i])['patch_save']
        #img = (np.mean(img,2)).astype(np.uint8)
        data_list.append(img)
        print(i,'/',len(img_list))
    return data_list

def gen_data_batch(data_list, is_training):
    sample_num = len(data_list)
    H = 256
    W = 256
    gt_all = torch.zeros([sample_num, 8, H, W]).cuda()
    for k in range(sample_num):
        img_original = data_list[k]
        #img_yuv = cv2.cvtColor(img_original, cv2.COLOR_BGR2YUV)
        #y, u, v = cv2.split(img_yuv)
        img = np.transpose(img_original,(2,0,1)) / 255.
        if is_training is True:
            angle = random.randint(0, 1) * 2
            img = np.rot90(img, angle, (1,2))
        gt = img.copy() # (50,216,192)
        gt_all[k] = torch.from_numpy(gt).float()


    return gt_all
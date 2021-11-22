from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from utils import generate_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
import argparse
import pytorch_ssim
from skimage.metrics import structural_similarity as SSIM
from models import*
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

test_path1 = r'./test_gray'
mask_path = r'./mask' # change the mask path to test any new masks which you can also generate by yourself.

mask_path=r'D:\siming_xihu\ziyang_pr\11.18fangzhen1\mask'

parser = argparse.ArgumentParser(description='Setting, compressive rate, size')
parser.add_argument('--B', default=8, type=int, help='compressive rate')
parser.add_argument('--size', default=[512, 512], type=int, help='input image resolution')
parser.add_argument('--num_block', default=26, type=int, help='number of reversible blocks')
args = parser.parse_args()
mask, mask_s = generate_masks(mask_path)

loss = nn.MSELoss()
loss.cuda()
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

def test_for_local(model):
    meas = scio.loadmat(r'D:\siming_xihu\ziyang_pr\11.18fangzhen1\meas_0.mat')['mask']
    meas = torch.from_numpy(meas).cuda().float()

    meas_re = torch.div(meas, mask_s)
    meas_re = torch.unsqueeze(meas_re, 1)

    maskt = mask.expand([1, 8, args.size[0], args.size[1]])
    Phi = maskt.cuda().float()
    Phi_s = torch.sum(Phi, 1)
    Phi_s[Phi_s == 0] = 1
    print(Phi.shape)
    out_save1 = torch.zeros([meas.shape[0], 8, args.size[0], args.size[1]]).cuda()

    for i in range(meas.shape[0]):
        with torch.no_grad():
            print(i)
            output = model(meas[i:i+1, :, :], Phi, Phi_s, meas_re, args)[-1]
            torch.cuda.synchronize()
            print(output.shape)
            out_save1[i, :, :, :] = output[0, :, :, :]
    scio.savemat(r'D:\siming_xihu\ziyang_pr\11.18fangzhen1\result.mat',{'pic': out_save1.detach().cpu().numpy()})
    for i in range(1):
        for j in range(8):
            cv2.imwrite(r'D:\siming_xihu\ziyang_pr\11.18fangzhen1\{}_{}.png'.format('result', j+10*i),out_save1.detach().cpu().numpy()[i,j, :, :]*255)

def test(test_path, result_path, model, args,ifsave=True):
    test_list = os.listdir(test_path)
    psnr_cnn = torch.zeros(len(test_list))
    ssim_cnn = torch.zeros(len(test_list))
    total_time=0
    total_num=0

    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])
        if "orig" in pic:
            pic = pic['orig']/255
        elif "patch_save" in pic:
            pic = pic['patch_save']/255

        pic_gt = np.zeros([pic.shape[2] // args.B, args.B, args.size[0], args.size[1]])
        for jj in range(pic.shape[2]):
            if jj % args.B == 0:
                meas_t = np.zeros([args.size[0], args.size[1]])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // args.B, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % args.B == 0 and jj != args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas).cuda().float()
        pic_gt = torch.from_numpy(pic_gt).cuda().float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        maskt = mask.expand([1, args.B, args.size[0], args.size[1]])
        # meas = meas.cuda().float()  # [batch,256 256]
        Phi = maskt.cuda().float()
        Phi_s = torch.sum(Phi, 1)
        Phi_s[Phi_s == 0] = 1

        out_save1 = torch.zeros([meas.shape[0], args.B, args.size[0], args.size[1]]).cuda()
        total_num+=meas.shape[0]

        with torch.no_grad():

            psnr_1 = 0
            ssim_1 = 0
            for ii in range(meas.shape[0]):
                # out_pic1=torch.zeros([1, args.B, args.size[0], args.size[1]]).cuda()
                start = time.time()
                ##################################### For large scale (2048,2048,8) without enugh GPU memory #################################################
                # part1 = model(meas[ii:ii + 1, ::][:, 0:1024, 0:1024], Phi[:, :, 0:1024, 0:1024], Phi_s[:, 0:1024, 0:1024], meas_re,args)[-1]
                # torch.cuda.synchronize()
                # print('part1')
                # part2 = model(meas[ii:ii + 1, ::][:, 1024:2048, 0:1024], Phi[:, :, 1024:2048, 0:1024], Phi_s[:, 1024:2048, 0:1024], meas_re,args)[-1]
                # torch.cuda.synchronize()
                # print('part2')
                # part3 = model(meas[ii:ii + 1, ::][:, 0:1024, 1024:2048], Phi[:, :, 0:1024, 1024:2048], Phi_s[:, 0:1024, 1024:2048], meas_re,args)[-1]
                # torch.cuda.synchronize()
                # print('part3')
                # part4 = model(meas[ii:ii + 1, ::][:, 1024:2048, 1024:2048], Phi[:, :, 1024:2048, 1024:2048], Phi_s[:, 1024:2048, 1024:2048], meas_re,args)[-1]
                # print('part4')
                # out_pic1[:,:,0:1024, 0:1024]=part1
                # out_pic1[:, :, 1024:2048, 0:1024] = part2
                # out_pic1[:, :, 0:1024, 1024:2048] = part3
                # out_pic1[:, :, 1024:2048, 1024:2048] = part4
                ######################################################################################################################################
                out_pic1 = model(meas[ii:ii + 1, ::], Phi, Phi_s, meas_re, args)[-1]
                torch.cuda.synchronize()
                end = time.time()
                total_time += (end - start)
                out_save1[ii, :, :, :] = out_pic1[0, :, :, :]
                for jj in range(args.B):
                    out_pic_forward = out_pic1[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                    mse_forward = mse_forward.data
                    psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)
                    ssim_1 += SSIM(out_pic_forward.cpu().numpy(),gt_t.cpu().numpy())

            psnr_1 = psnr_1 / (meas.shape[0] * args.B)
            ssim_1 = ssim_1 / (meas.shape[0] * args.B)
            psnr_cnn[i] = psnr_1
            ssim_cnn[i] = ssim_1

            if ifsave:
                a = test_list[i]
                name1 = result_path + '/cnn_' + a[0:len(a) - 4] + '_{:.4f}'.format(psnr_1) + '.mat'
                out_save1 = out_save1.cpu()
                scio.savemat(name1, {'pic': out_save1.numpy()})
    print("average psnr result: {:.4f}".format(torch.mean(psnr_cnn)))
    print("average ssim result: {:.4f}".format(torch.mean(ssim_cnn)))
    print("average time: {:.4f}".format(total_time/total_num))
    print("meas numeber: {:.4f}".format(total_num))

if __name__ == '__main__':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_model=GAP_net(args).cuda()
    model_save_filename = 'base_3'
    test_model_state = torch.load('./model/' + model_save_filename + "/model_state_3_stage.pth")
    test_model.load_state_dict(test_model_state,strict=False)
    # test(test_path1, result_path, test_model.eval(), args)
    #  ./model/base_2/model_state_2_stage.pth  ---------------two   stage  (Some codes need to be commented out in the file of 'model.py' to convert to '2-stage' model.)
    #  ./model/base_3/model_state_3_stage.pth  ---------------three stage
    #  The model is trained on NVIDIA A40, the results on other types of GPU may cause a fluctuation on PSNR with in 0.1dB.
    test_for_local(test_model.eval())

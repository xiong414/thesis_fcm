# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-14 12:13:07
@Email: xiong3219@icloud.com
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 正方+圆 人工图像
    # pic_shape = [40, 40]
    # pic = np.zeros(shape=pic_shape)
    # for i in range(pic_shape[0]):
    #     for j in range(pic_shape[1]):
    #         if i < pic_shape[0] / 2 and j < pic_shape[1] / 2:
    #             pic[i][j] = 0
    #         if i >= pic_shape[0] / 2 and j < pic_shape[1] / 2:
    #             pic[i][j] = 2
    #         if i < pic_shape[0] / 2 and j >= pic_shape[1] / 2:
    #             pic[i][j] = 3
    #         if i >= pic_shape[0] / 2 and j >= pic_shape[1] / 2:
    #             pic[i][j] = 1
    #         if (i - pic_shape[0] / 2)**2 + (j - pic_shape[0] / 2)**2 < (
    #                 pic_shape[0] / 4)**2:
    #             pic[i][j] = 4
    # plt.imsave('pic/artif_pic_penta_3.png', pic, cmap=plt.cm.gray)

    # 对称二分类图
    # pic_shape = [64, 64]
    # pic = np.zeros(shape=pic_shape)
    # for i in range(pic_shape[0]):
    #     for j in range(pic_shape[1]):
    #         if j < pic_shape[1] / 2:
    #             pic[i][j] = 0
    #         else:
    #             pic[i][j] = 1
    # plt.imsave('pic/artif_pic_syn_2.png', pic, cmap=plt.cm.gray)

    # 三分类图
    pic_shape = [35, 35]
    pic = np.zeros(shape=pic_shape)
    for i in range(pic_shape[0]):
        for j in range(pic_shape[1]):
            if j < pic_shape[1] / 2:
                if i + j < pic_shape[1]:
                    pic[i][j] = 1
                else:
                    pic[i][j] = 2
            else:
                if i > j:
                    pic[i][j] = 2
    plt.imsave('pic/artif_pic_tri_3.png', pic, cmap=plt.cm.gray)


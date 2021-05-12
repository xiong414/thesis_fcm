# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-05-10 20:29:19
@Email: xiong3219@icloud.com
'''

import cv2 as cv
import numpy as np
from clustering import gaussian_noise

# np.set_printoptions(threshold = np.inf) 

def neighbor(x, y, img_shape):
    n_coor = []
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if i != x or j != y:
                if i in [_ for _ in range(0, img_shape[0])] and j in [__ for __ in range(0, img_shape[1])]:
                    n_coor.append([i, j])
    return n_coor

# def get_value(n, img):
#     val = np.zeros()
#     for point in n:


if __name__ == '__main__':
    # pic_addr = 'pic/artif_pic_tri_2.png'
    pic_addr = '/Users/xiongguoqing/Documents/学习资料-本科/毕业论文/代码/FCM/pic_output/20210511214653/FLICM_SW_0_30__96.41_41.54.png'
    pic = cv.imread(pic_addr, 0)
    pic_shape = np.shape(pic)
    # pic_noise = gaussian_noise(pic, mu=0, sigma=1)
    # vars = []
    # for i in range(pic_shape[0]):
    #     for j in range(pic_shape[1]):
    #         n = neighbor(i, j, pic_shape)
    #         v = []
    #         for _ in n:
    #             v.append(pic_noise[_[0], _[1]])
    #         # print(v)
    #         # print(np.var(v))
    #         vars.append(np.var(v))
    #     #     break
    #     # break
    # print(np.sqrt(np.var(pic)), np.sqrt(np.var(pic_noise)))
    # print(np.sqrt(np.mean(vars)))
    print(pic)
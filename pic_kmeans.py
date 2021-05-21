# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-05-18 21:46:37
@Email: xiong3219@icloud.com
'''
from sklearn.cluster import KMeans
import  clustering
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def repaint_pic(img, img_ref, centers):
    pic = []
    ref_set = list(set(np.reshape(img_ref, (np.size(img_ref)))))
    ref_set.sort()
    centers_arrange = centers.tolist()
    centers_arrange.sort()
    for i in img:
        pixel = centers[i]
        std_index = centers_arrange.index(pixel)
        pixel_std = ref_set[std_index]
        pic.append(pixel_std)
    return pic

def save_pic(pic, mu, sigma, sa, psnr, start_time):
    addr = 'pic_output/'
    mu_ = str(mu)
    sigma_ = str(sigma)
    sa_ = '{:.2f}'.format(sa*100)
    psnr_ = '{:.2f}'.format(psnr)
    name = 'KMeans_' + mu_ + '_' + sigma_ + '__' + sa_ + '_' + psnr_ + '.png'
    plt.imsave(addr + start_time + '/' + name, pic, cmap='gray')
    print('image saved!')


def main(addr, mu, sigma, c, start_time):
    pic = cv.imread(addr, 0)
    pic_noise = clustering.gaussian_noise(pic, mu=mu, sigma=sigma)
    pic_noise_array = np.reshape(pic_noise, np.size(pic)).reshape(-1, 1)
    cluster_num = c

    clf = KMeans(n_clusters=cluster_num)
    clf.fit(pic_noise_array)

    centers = clf.cluster_centers_
    labels = clf.labels_

    pic_output = repaint_pic(img=labels, img_ref=pic, centers = centers)
    pic_output = np.reshape(pic_output, np.shape(pic))

    sa = clustering.SA(img_ref=pic, img=pic_output)
    psnr = clustering.PSNR(img_ref=pic, img=pic_output)

    print('SA: \t{:.2f} %'.format(sa * 100))
    print('PSNR:\t{:.2f}'.format(psnr))

    save_pic(pic=pic_output, mu=mu, sigma=sigma, sa=sa, psnr=psnr, start_time=start_time)

    # plt.subplot(1, 3, 1)
    # plt.imshow(pic, cmap=plt.cm.gray)
    # plt.subplot(1, 3, 2)
    # plt.imshow(pic_noise, cmap=plt.cm.gray)
    # plt.subplot(1, 3, 3)
    # plt.imshow(pic_output, cmap=plt.cm.gray)
    # plt.show()
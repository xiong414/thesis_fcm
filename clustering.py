# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-28 14:54:46
@Email: xiong3219@icloud.com
'''

import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def gaussian_noise(image, mu, sigma):
    image_output = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            noise = np.random.normal(loc=mu, scale=sigma)
            if int(image[i][j] + noise) > 255:
                image_output[i][j] = int(image[i][j] - noise)
            else:
                image_output[i][j] = int(image[i][j] + noise)
    return image_output


def repaint_pic(X, membership, centroids, img_ref=None):
    pic = []
    if img_ref is None:
        for x, m in zip(X, membership):
            pic.append(centroids[np.argmax(m)])
    else:
        ref_set = list(set(np.reshape(img_ref, (np.size(img_ref)))))
        ref_set.sort()
        centroids_arrange = centroids.tolist()
        centroids_arrange.sort()
        for x, m in zip(X, membership):
            pixel = centroids[np.argmax(m)]
            std_index = centroids_arrange.index(pixel)
            pixel_std = ref_set[std_index]
            pic.append(pixel_std)
    return pic


def SA(img_ref, img):
    sa = 0
    fenzi = 0
    fenmu = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == img_ref[i][j]:
                fenzi += 1
            fenmu += 1
    return fenzi / fenmu


def PSNR(img_ref, img):
    M, N = np.shape(img_ref)
    MSE = 0
    for i in range(M):
        for j in range(N):
            MSE += np.linalg.norm(float(img_ref[i][j]) - float(img[i][j]))
    MSE = MSE / (M * N)
    return 10 * np.log10(255**2 / MSE)


def display(model, plot=False):
    print('=' * 7, 'INDEX', '=' * 7)
    print('SA: \t{:.2f} %'.format(model.sa * 100))
    print('PSNR:\t{:.2f}'.format(model.psnr))
    print('time:\t{:.2f} s'.format(model.total_time))
    print('speed:\t{:.5f} r/s'.format(model._iter / model.total_time))

    if plot == True:
        plt.subplot(1, 3, 1)
        plt.imshow(model.pic, cmap=plt.cm.gray)
        plt.subplot(1, 3, 2)
        plt.imshow(model.pic_input, cmap=plt.cm.gray)
        plt.subplot(1, 3, 3)
        plt.imshow(model.pic_output, cmap=plt.cm.gray)
        plt.show()


def build_model(func):
    def wrapper(img_addr, sigma, mu=0, epsilon=0.1, iter_max=100, **kwargs):
        start_time = time.time()
        img = cv.imread(img_addr, 0)
        img_array = np.reshape(img, (np.size(img), 1))
        img_noise = gaussian_noise(img, mu=mu, sigma=sigma)
        img_noise_array = np.reshape(img_noise, (np.size(img), 1))
        model = func(x=img_noise_array, img_shape=np.shape(img), **kwargs)
        model.init_para()
        c_list = []
        for i in range(0, iter_max):
            model.iter_centroids()
            model.iter_membership()
            c_list.append(model.centroids)
            if i < 1:
                print('{:3}'.format(i))
            else:
                diff = np.linalg.norm((c_list[i] - c_list[i - 1]), 2)
                print('{:3}| {:.5f}'.format(i, diff))
                if diff <= epsilon:
                    break
        pic = np.reshape(
            repaint_pic(X=img_array,
                        membership=model.membership_mat,
                        centroids=model.centroids,
                        img_ref=img), np.shape(img))
        sa = SA(img_ref=img, img=pic)
        psnr = PSNR(img_ref=img, img=pic)
        total_time = time.time() - start_time
        m = Model(model=model,
                  pic=img,
                  pic_input=img_noise,
                  pic_output=pic,
                  sa=sa,
                  psnr=psnr,
                  total_time=total_time,
                  _iter=len(c_list))
        return m

    return wrapper


class Model:
    def __init__(self, model, pic, pic_input, pic_output, sa, psnr, total_time,
                 _iter):
        self.model = model
        self.pic = pic
        self.pic_input = pic_input
        self.pic_output = pic_output
        self.sa = sa
        self.psnr = psnr
        self.total_time = total_time
        self._iter = _iter


class FCM:
    def __init__(self, x, m, cluster_num, img_shape):
        self.x = x
        self.m = m
        self.cluster_num = cluster_num
        self.membership_mat = []
        self.centroids = []
        self.distance_mat = []
        self.img_shape = img_shape

    def init_para(self):
        membership_mat = np.random.random((len(self.x), self.cluster_num))
        membership_mat = np.divide(
            membership_mat,
            np.sum(membership_mat, axis=1)[:, np.newaxis])
        self.membership_mat = membership_mat
        membership_mat_m = membership_mat**self.m
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids

    def iter_centroids(self):
        membership_mat_m = self.membership_mat**self.m
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                new_membership_mat[i][j] = 1. / np.sum((distance_mat[i][j] / distance_mat[i])**(2 / (self.m - 1)))
        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat


class FLICM(FCM):
    def __init__(self, x, cluster_num, m, img_shape):
        self.x = x
        self.cluster_num = cluster_num
        self.m = m
        self.distance_mat = []
        self.centroids = []
        self.membership_mat = []
        self.img_shape = img_shape

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))

        # distance matrix
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)

        # g factor
        def g(k, i):
            val = 0
            i_x, i_y = i // self.img_shape[1], i % self.img_shape[1]
            i_coo = [i_x, i_y]
            neighbor_coordinate = []
            neighbor_array = []
            if i_coo[0] % self.img_shape[0] != 0 and i_coo[1] % self.img_shape[1] != 0:
                if i_coo[0] % (self.img_shape[0] - 1) != 0 and i_coo[1] % (self.img_shape[1] - 1) != 0:
                    neighbor_coordinate.append([i_x - 1, i_y - 1])
                    neighbor_coordinate.append([i_x - 1, i_y])
                    neighbor_coordinate.append([i_x - 1, i_y + 1])
                    neighbor_coordinate.append([i_x, i_y - 1])
                    neighbor_coordinate.append([i_x, i_y + 1])
                    neighbor_coordinate.append([i_x + 1, i_y - 1])
                    neighbor_coordinate.append([i_x + 1, i_y])
                    neighbor_coordinate.append([i_x + 1, i_y + 1])
            for n in neighbor_coordinate:
                neighbor_array.append(n[0] * self.img_shape[1] + n[1])
            for j_coo, j_arr in zip(neighbor_coordinate, neighbor_array):
                djk = 1. / (np.linalg.norm(np.array(j_coo) - np.array(i_coo)))
                uikm = ((1 - self.membership_mat[j_arr][k])**self.m)
                xkvi = (np.linalg.norm(self.x[j_arr] - self.centroids[k], 2))
                val += djk * uikm * xkvi
            return val

        # membership matrix
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                if i % self.img_shape[1] != 0 or i >= self.img_shape[1] or i % (self.img_shape[1] + 1) != 0 or i < (len(self.x) - self.img_shape[1]):
                    new_membership_mat[i][j] = 1. / np.sum(((distance_mat[i][j]**2 + g(j, i)) / ((distance_mat[i]**2) + [g(r, i)for r in range(len(self.centroids))])) ** (1 / (self.m - 1)))
                else:
                    new_membership_mat[i][j] = 1. / np.sum((distance_mat[i][j]**2 / (distance_mat[i]**2))**(1 / (self.m - 1)))
        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat

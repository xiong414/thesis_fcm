# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-04 15:03:36
@Email: xiong3219@icloud.com
'''

import multiprocessing
import copy
import math
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(addr, c):
    X = []
    with open(addr, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            X.append([float(row[i]) for i in range(c)])
    return X


def read_label(addr):
    label_class = []
    label = []
    with open(addr, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[-1] not in label_class:
                label_class.append(row[-1])
            label.append(label_class.index(row[-1]))
    return label_class, label


class FCM(object):
    def __init__(self, x, c_clusters, m, record_path=False):
        self.x = x
        self.c_clusters = c_clusters
        self.m = m
        self.membership_mat = []
        self.centroids = []
        self.distance_mat = []
        self.record_path = record_path
        self.centroids_path = []
        self.pool = multiprocessing.Pool(processes=2)

    def init_para(self):
        # initialize membership matrix
        # 此处生成的隶属度矩阵是n*c，并非c*n
        membership_mat = np.random.random((len(self.x), self.c_clusters))
        membership_mat = np.divide(
            membership_mat,
            np.sum(membership_mat, axis=1)[:, np.newaxis])
        self.membership_mat = membership_mat
        # initialize centroids
        membership_mat_m = membership_mat**self.m
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids
        # initialize centroids_paths
        if self.record_path is True:
            for _ in range(self.c_clusters):
                self.centroids_path.append([])

    def iter_centroids(self):
        # add paths
        if self.record_path is True:
            for i, _ in enumerate(self.centroids):
                self.centroids_path[i].append(self.centroids[i])
        # iter centroids
        membership_mat_m = self.membership_mat**self.m
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))

        # D.T
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)

        # U.T

        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                new_membership_mat[i][j] = 1. / np.sum(
                    (distance_mat[i][j] / distance_mat[i])**(2 / (self.m - 1)))

        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat

    def calc_obj_function(self):
        obj_function = 0
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                obj_function += ((self.membership_mat[i][j]**self.m) *
                                 (self.distance_mat[i][j]**2))
        return obj_function


class PCM(object):
    def __init__(self, x, c_clusters, m):
        self.x = x
        self.c_clusters = c_clusters
        self.m = m
        self.membership_mat = []
        self.centroids = []
        self.distance_mat = []
        self.centroids_path = []
        self.theta = np.zeros((c_clusters))

    def init_para(self):
        membership_mat = np.random.random((len(self.x), self.c_clusters))
        membership_mat = np.divide(
            membership_mat,
            np.sum(membership_mat, axis=1)[:, np.newaxis])
        self.membership_mat = membership_mat
        membership_mat_m = membership_mat**self.m
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids
        for _ in range(self.c_clusters):
            self.centroids_path.append([])

    def iter_centroids(self):
        for i, _ in enumerate(self.centroids):
            self.centroids_path[i].append(self.centroids[i])
        membership_mat_m = self.membership_mat**self.m
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))
        membership_mat_m = self.membership_mat**self.m
        for i, c in enumerate(self.centroids):
            fenzi, fenmu = 0, 0
            for j, x in enumerate(self.x):
                fenzi += membership_mat_m[j][i] * (distance_mat[j][i]**2)
                fenmu += membership_mat_m[j][i]
            self.theta[i] = fenzi / fenmu
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                new_membership_mat[i][j] = 1. / (1 + (
                    (distance_mat[i][j]**2) / self.theta[j])**(1 /
                                                               (self.m - 1)))
        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat

    def calc_obj_function(self):
        obj_function = 0
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                obj_function += (self.membership_mat[i][j]**
                                 self.m) * (self.distance_mat[i][j]**2)
        penalty = 0
        for i, c in enumerate(self.centroids):
            penalty += self.theta[i] * np.sum([(1 - self.membership_mat[j][i])
                                               **self.m
                                               for j in range(len(self.x))])
        return obj_function + penalty


class PFCM(object):
    def __init__(self, x, c_clusters, m, n, a, b):
        self.x = x
        self.c_clusters = c_clusters
        self.m = m
        self.n = n
        self.a = a
        self.b = b
        self.u_membership_mat = []
        self.t_membership_mat = []
        self.centroids = []
        self.centroids_path = []
        self.distance_mat = []
        self.theta = np.zeros((c_clusters))

    def init_para(self):
        # initialize u membership matrix
        u_membership_mat = np.random.random((len(self.x), self.c_clusters))
        u_membership_mat = np.divide(
            u_membership_mat,
            np.sum(u_membership_mat, axis=1)[:, np.newaxis])
        self.u_membership_mat = u_membership_mat
        # initialize t membership matrix
        t_membership_mat = np.random.random((len(self.x), self.c_clusters))
        t_membership_mat = np.divide(
            t_membership_mat,
            np.sum(t_membership_mat, axis=1)[:, np.newaxis])
        self.t_membership_mat = t_membership_mat
        # initialize centroids
        membership_mat_m = self.a * u_membership_mat**self.m + self.b * t_membership_mat**self.n
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids
        # initialize centroids_paths
        for _ in range(self.c_clusters):
            self.centroids_path.append([])

    def iter_centroids(self):
        # add paths
        for i, _ in enumerate(self.centroids):
            self.centroids_path[i].append(self.centroids[i])
        # iter centroids
        membership_mat_m = self.a * self.u_membership_mat**self.m + self.b * self.t_membership_mat**self.n
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.x),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.u_membership_mat))
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)
        new_u_membership_mat = np.zeros(shape=np.shape(self.u_membership_mat))
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                new_u_membership_mat[i][j] = 1. / np.sum(
                    (distance_mat[i][j] / distance_mat[i])**(2 / (self.m - 1)))
        self.u_membership_mat = new_u_membership_mat

        new_t_membership_mat = np.zeros(shape=np.shape(self.t_membership_mat))
        t_membership_mat_m = self.t_membership_mat**self.n
        for i, c in enumerate(self.centroids):
            fenzi, fenmu = 0, 0
            for j, x in enumerate(self.x):
                fenzi += t_membership_mat_m[j][i] * (distance_mat[j][i]**2)
                fenmu += t_membership_mat_m[j][i]
            self.theta[i] = fenzi / fenmu
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                new_t_membership_mat[i][j] = 1. / (1 + (
                    (distance_mat[i][j]**2) / self.theta[j])**(1 /
                                                               (self.m - 1)))
        self.t_membership_mat = new_t_membership_mat
        self.distance_mat = distance_mat

    def calc_obj_function(self):
        obj_function = 0
        for i, x in enumerate(self.x):
            for j, c in enumerate(self.centroids):
                obj_function += (self.a * self.u_membership_mat[i][j]**self.m +
                                 self.b * self.t_membership_mat[i][j]**self.n
                                 ) * (self.distance_mat[i][j]**2)
        penalty = 0
        for i, c in enumerate(self.centroids):
            penalty += self.theta[i] * np.sum(
                [(1 - self.t_membership_mat[j][i])**self.n
                 for j in range(len(self.x))])
        return obj_function + penalty


class FLICM(object):
    def __init__(self, X, c_clusters, m, img_shape):
        self.X = X
        self.c_clusters = c_clusters
        self.m = m
        self.distance_mat = []
        self.centroids = []
        self.membership_mat = []
        self.img_shape = img_shape

    def init_para(self):
        membership_mat = np.random.random((len(self.X), self.c_clusters))
        membership_mat = np.divide(
            membership_mat,
            np.sum(membership_mat, axis=1)[:, np.newaxis])
        self.membership_mat = membership_mat
        # initialize centroids
        membership_mat_m = membership_mat**self.m
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.X),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids

    def iter_centroids(self):
        membership_mat_m = self.membership_mat**self.m
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.X),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))

        # distance matrix
        for i, x in enumerate(self.X):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)

        # g factor
        def g(k, i):
            val = 0
            i_x, i_y = i // self.img_shape[1], i % self.img_shape[1]
            i_coo = [i_x, i_y]
            neighbor_coordinate = []
            neighbor_array = []
            if i_coo[0] % self.img_shape[0] != 0 and i_coo[1] % self.img_shape[
                    1] != 0:
                if i_coo[0] % (self.img_shape[0] - 1) != 0 and i_coo[1] % (
                        self.img_shape[1] - 1) != 0:
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
                xkvi = (np.linalg.norm(self.X[j_arr] - self.centroids[k], 2))
                val += djk * uikm * xkvi
            return val

        # membership matrix
        for i, x in enumerate(self.X):
            for j, c in enumerate(self.centroids):
                if i % self.img_shape[1] != 0 or i >= self.img_shape[
                        1] or i % (self.img_shape[1] + 1) != 0 or i < (
                            len(self.X) - self.img_shape[1]):
                    new_membership_mat[i][j] = 1. / np.sum(
                        ((distance_mat[i][j]**2 + g(j, i)) /
                         ((distance_mat[i]**2) +
                          [g(r, i) for r in range(len(self.centroids))]))
                        **(1 / (self.m - 1)))
                else:
                    new_membership_mat[i][j] = 1. / np.sum(
                        (distance_mat[i][j]**2 /
                         (distance_mat[i]**2))**(1 / (self.m - 1)))

        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat


class BF_FLICM(object):
    def __init__(self, X, c_clusters, m, img_shape, sigma_d, sigma_r):
        self.X = X
        self.c_clusters = c_clusters
        self.m = m
        self.distance_mat = []
        self.centroids = []
        self.membership_mat = []
        self.img_shape = img_shape
        self.X_matrix = np.reshape(self.X, img_shape)
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.pool = multiprocessing.Pool(processes=2)

    def init_para(self):
        membership_mat = np.random.random((len(self.X), self.c_clusters))
        membership_mat = np.divide(
            membership_mat,
            np.sum(membership_mat, axis=1)[:, np.newaxis])
        self.membership_mat = membership_mat
        # initialize centroids
        membership_mat_m = membership_mat**self.m
        centroids = np.divide(
            np.dot(membership_mat_m.T, self.X),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = centroids

    def iter_centroids(self):
        membership_mat_m = self.membership_mat**self.m
        new_centroids = np.divide(
            np.dot(membership_mat_m.T, self.X),
            np.sum(membership_mat_m.T, axis=1)[:, np.newaxis])
        self.centroids = new_centroids

    def iter_membership(self):
        distance_mat = np.zeros(shape=np.shape(self.membership_mat))
        new_membership_mat = np.zeros(shape=np.shape(self.membership_mat))

        # distance matrix
        for i, x in enumerate(self.X):
            for j, c in enumerate(self.centroids):
                distance_mat[i][j] = np.linalg.norm(x - c, 2)

        # g factor
        def g(k, i):
            val = 0
            i_x, i_y = i // self.img_shape[1], i % self.img_shape[1]
            i_coo = [i_x, i_y]
            neighbor_coordinate = []
            neighbor_array = []
            if i_coo[0] % self.img_shape[0] != 0 and i_coo[1] % self.img_shape[
                    1] != 0:
                if i_coo[0] % (self.img_shape[0] - 1) != 0 and i_coo[1] % (
                        self.img_shape[1] - 1) != 0:
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
                exp_spatial = np.exp(-(
                    np.linalg.norm(np.array([j_coo]) - np.array([i_coo]), 2) /
                    2 / self.sigma_d**2))
                exp_similar = np.exp(
                    -(np.linalg.norm(self.X[j_arr] - self.X[i], 2) / 2 /
                      self.sigma_r**2))
                uikm = ((1 - self.membership_mat[j_arr][k])**self.m)
                xkvi = (np.linalg.norm(self.X[j_arr] - self.centroids[k], 2))
                val += exp_spatial * exp_similar * uikm * xkvi
            return val

        self.pool.apply_async(g, (1, ))

        # membership matrix
        def i():
            for i, x in enumerate(self.X):
                for j, c in enumerate(self.centroids):
                    if i % self.img_shape[1] != 0 or i >= self.img_shape[
                            1] or i % (self.img_shape[1] + 1) != 0 or i < (
                                len(self.X) - self.img_shape[1]):
                        new_membership_mat[i][j] = 1. / np.sum(
                            ((distance_mat[i][j]**2 + g(j, i)) /
                             ((distance_mat[i]**2) +
                              [g(r, i) for r in range(len(self.centroids))]))
                            **(1 / (self.m - 1)))
                    else:
                        new_membership_mat[i][j] = 1. / np.sum(
                            (distance_mat[i][j]**2 /
                             (distance_mat[i]**2))**(1 / (self.m - 1)))

        self.pool.apply_async(i, (2, ))
        i()

        self.distance_mat = distance_mat
        self.membership_mat = new_membership_mat


# for numeric
class membership2class(object):
    def __init__(self, membership_mat):
        self.membership_mat = membership_mat
        self.label = []

    def convert2label(self):
        # fix membership_mat
        # 以前5个为基准
        old_membership_mat = self.membership_mat
        new_membership_mat = []
        for column in range(np.shape(self.membership_mat)[1]):
            top = []
            for i in old_membership_mat:
                raw_label = list(i).index(np.max(i))
                top.append(raw_label)
                if len(top) == 5:
                    break
            fix_flag = max(top, key=top.count)
            new_membership_mat.append(old_membership_mat[:, fix_flag])
            old_membership_mat = np.delete(old_membership_mat,
                                           fix_flag,
                                           axis=1)
        new_membership_mat = np.array(new_membership_mat).T
        for i in new_membership_mat:
            raw_label = list(i).index(np.max(i))
            self.label.append(raw_label)


class clf_index(object):
    def __init__(self, label, predict):
        self.label = label
        self.predict = predict
        self.acc_rate = 0
        self.calc_acc_rate()

    def calc_acc_rate(self):
        for l, p in zip(self.label, self.predict):
            if l is p:
                self.acc_rate += 1
        self.acc_rate /= len(self.label)


# only available for 2D dataset
class draw(object):
    def __init__(self, X, U, V, V_path):
        self.X = X
        self.U = U
        self.V = V
        self.V_path = V_path
        self.color_array = []

    def plot(self):
        # 给不同的类型分配颜色
        # plot V
        for v in self.V:
            color = tuple(float(x) for x in np.random.rand(3, 1))
            self.color_array.append(color)
            plt.plot(v[0], v[1], '*', color=color)

        # plot X based on U
        for x, u in zip(self.X, self.U):
            member = list(u).index(np.max(u))
            plt.plot(x[0], x[1], '.', color=self.color_array[member])

        # plot V path
        for path in self.V_path:
            x_, y_ = [], []
            for p in path:
                x_.append(p[0])
                y_.append(p[1])
                plt.scatter(p[0], p[1], c='b', s=0.3)
            plt.plot(x_, y_, 'b-', linewidth='0.2')

        plt.show()


# for pic clustering
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


# 高斯噪音
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


# 椒盐噪音(noise_rate为噪音率，可调节参数)
def saltandpepper_noise(image, noise_rate):
    new_image = copy.deepcopy(image)
    noise_matrix = np.random.randint(0, 100, size=image.shape)
    for i in range(len(noise_matrix)):
        for j in range(len(noise_matrix[i])):
            if noise_matrix[i][j] > 99 - noise_rate:
                new_image[i][j] = 255
            elif noise_matrix[i][j] < noise_rate:
                new_image[i][j] = 0
    return new_image


# 分割指标SA
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


# 分割指标PSNR
def PSNR(img_ref, img):
    M, N = np.shape(img_ref)
    MSE = 0
    for i in range(M):
        for j in range(N):
            MSE += np.linalg.norm(float(img_ref[i][j]) - float(img[i][j]))
    MSE = MSE / (M * N)
    return 10 * np.log10(255**2 / MSE)

# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-28 15:09:59
@Email: xiong3219@icloud.com
'''
import sys
import time

from numpy import add
import clustering
import pic_kmeans

if __name__ == '__main__':
    try:
        arg = sys.argv[1]
        p = False
    except:
        p = True

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    strtime = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    print(start_time)

    img_addr = 'pic/artif_pic_syn_1.png'

    mu = 0
    sigma = 50
    c = 2

    model1 = clustering.build_model(clustering.FCM, progress=p)(img_addr=img_addr, mu=mu, sigma=sigma, cluster_num=c, m=2)
    clustering.display(model1, save_pic=1, start_time=strtime, plot=0, save_noise=1)

    model_k = pic_kmeans.main(addr=img_addr, mu=mu, sigma=sigma, c=c, start_time=strtime)

    # model2 = clustering.build_model(clustering.FLICM, progress=p)(img_addr=img_addr, mu=mu, sigma=sigma, cluster_num=c, m=2)
    # clustering.display(model2, save_pic=1, start_time=strtime, plot=0)

    # model3 = clustering.build_model(clustering.FLSICM, progress=p)(img_addr=img_addr, mu=mu, sigma=sigma, cluster_num=c, m=2)
    # clustering.display(model3, save_pic=1, start_time=strtime, plot=0)

    # model4 = clustering.build_model(clustering.WFLSICM, progress=p)(img_addr=img_addr, mu=mu, sigma=sigma, cluster_num=c, m=2)
    # clustering.display(model4, save_pic=1, start_time=strtime, plot=0)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(end_time)

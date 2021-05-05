# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-28 15:09:59
@Email: xiong3219@icloud.com
'''
import sys
import time
import clustering

if __name__ == '__main__':
    try:
        arg = sys.argv[1]
        p = False
    except:
        p = True

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    strtime = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    print(start_time)

    # img_addr = 'pic/3096.png'
    # gt = 'pic/3096_gt.png'
    img_addr = 'pic/artif_pic_syn_1.png'

    model1 = clustering.build_model(clustering.FCM, progress=p)(img_addr=img_addr, mu=70, sigma=80, cluster_num=2, m=2, epsilon=0.001)
    clustering.display(model1, save_pic=1, start_time=strtime, plot=0)

    model2 = clustering.build_model(clustering.FLICM, progress=p)(img_addr=img_addr, mu=70, sigma=80, cluster_num=2, m=2, epsilon=0.001)
    clustering.display(model2, save_pic=1, start_time=strtime, plot=0)

    model3 = clustering.build_model(clustering.FLICM_S, progress=p)(img_addr=img_addr, mu=70, sigma=80, cluster_num=2, m=2, epsilon=0.001)
    clustering.display(model3, save_pic=1, start_time=strtime, plot=0)

    model4 = clustering.build_model(clustering.FLICM_SD, progress=p)(img_addr=img_addr, mu=70, sigma=80, cluster_num=2, m=2, epsilon=0.001)
    clustering.display(model4, save_pic=1, start_time=strtime, plot=0)  

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(end_time)

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
    print(start_time)

    img_addr = 'pic/artif_pic_2.png'
    model1 = clustering.build_model(clustering.FCM, progress=p)(img_addr=img_addr, mu=0, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model1, save_pic=True)

    model2 = clustering.build_model(clustering.FLICM, progress=p)(img_addr=img_addr, mu=0, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model2, save_pic=True)

    # model3 = clustering.build_model(clustering.BFFLICM, progress=p)(img_addr=img_addr, mu=0, sigma=40, cluster_num=5, m=2, epsilon=0.01, sigma_d=20, sigma_r=300)
    # clustering.display(model3, save_pic=True)

    model4 = clustering.build_model(clustering.FLICM_1, progress=p)(img_addr=img_addr, mu=0, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model4, save_pic=True)

    model5 = clustering.build_model(clustering.FLICM_2, progress=p)(img_addr=img_addr, mu=0, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model5, save_pic=True)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(end_time)

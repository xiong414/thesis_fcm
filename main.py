# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-28 15:09:59
@Email: xiong3219@icloud.com
'''
import clustering

if __name__ == '__main__':
    img_addr = 'pic/artif_pic.png'
    model1 = clustering.build_model(clustering.FCM)(img_addr=img_addr, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model1)

    # model2 = clustering.build_model(clustering.FLICM)(img_addr=img_addr, sigma=20, cluster_num=5, m=2, epsilon=0.01)
    # clustering.display(model2)

    # model3 = clustering.build_model(clustering.BFFLICM)(img_addr=img_addr, sigma=20, cluster_num=5, m=2, epsilon=0.01, sigma_d=50, sigma_r=40)
    # clustering.display(model3)

# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-28 15:09:59
@Email: xiong3219@icloud.com
'''
import clustering

if __name__ == '__main__':
    img_addr = 'pic/artif_pic_3.png'
    model1 = clustering.build_model(clustering.FCM)(img_addr=img_addr, mu=0, sigma=40, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model1, save_pic=False)

    model2 = clustering.build_model(clustering.FLICM)(img_addr=img_addr, mu=0, sigma=40, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model2, save_pic=False)

    # model3 = clustering.build_model(clustering.BFFLICM)(img_addr=img_addr, mu=0, sigma=40, cluster_num=5, m=2, epsilon=0.01, sigma_d=20, sigma_r=300)
    # clustering.display(model3, save_pic=False)

    model4 = clustering.build_model(clustering.FLICM_1)(img_addr=img_addr, mu=0, sigma=40, cluster_num=5, m=2, epsilon=0.01)
    clustering.display(model4, save_pic=False)

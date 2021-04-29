# !/usr/bin/env python3
# coding=utf-8
'''
@Author: Xiong Guoqing
@Date: 2021-04-11 15:56:26
@Email: xiong3219@icloud.com
'''
import csv
relative_path = '/Users/xiongguoqing/Library/Mobile Documents/com~apple~CloudDocs/Documents/学习资料-本科/毕业论文/代码/FCM/'
with open(relative_path + 'glass.data', 'r', encoding='utf-8') as f:
    with open(relative_path + 'glass_fix.data', 'w', encoding='utf-8') as ff:
        reader = csv.reader(f)
        writer = csv.writer(ff)
        for row in reader:
            new_row = row[1:]
            for i, element in enumerate(new_row):
                new_row[i] = float(element)
                if i == len(new_row) - 1:
                    new_row[i] = int(element)
            writer.writerow(new_row)

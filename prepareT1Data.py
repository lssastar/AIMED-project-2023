import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shutil

os.chdir('D:/vscodes/AIinMed/final_project')  # 设置工作路径为当前路径

# 读取csv文件
df = pd.read_csv('./data/matadata/ADNI2T1_12_27_2023.csv')
# print(df.head())

# 数据保存位置
src_path = 'D:\\Data\\ADNI2T1\\ADNI'
out_path = 'D:\\Data\\ADNI2T1\\prepared_data'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# 获取src_path下所有文件夹的名称
folders = os.listdir(src_path)

# 将原始数据按照df中的Subject和Research Group进行分类
for folder in folders:
    # print(folder)
    folder_path = os.path.join(src_path, folder)
    if folder in df['Subject'].values:
        # print(folder)
        # print(df[df['Subject'] == folder]['Research group'].values[0])
        # 检查folder对应的Description是否为localizer
        if df[df['Subject'] == folder]['Description'].values[0] != 'localizer':
            continue
        if df[df['Subject'] == folder]['Group'].values[0] == 'AD':
            ad_path = os.path.join(out_path, 'AD')
            if not os.path.exists(ad_path):
                os.makedirs(ad_path)
            dest_path = os.path.join(ad_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))

        elif df[df['Subject'] == folder]['Group'].values[0] == 'CN':
            cn_path = os.path.join(out_path, 'CN')
            if not os.path.exists(cn_path):
                os.makedirs(cn_path)
            dest_path = os.path.join(cn_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))
        elif df[df['Subject'] == folder]['Group'].values[0] == 'MCI':
            mci_path = os.path.join(out_path, 'MCI')
            if not os.path.exists(mci_path):
                os.makedirs(mci_path)
            dest_path = os.path.join(mci_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))
        elif df[df['Subject'] == folder]['Group'].values[0] == 'LMCI':
            lmci_path = os.path.join(out_path, 'LMCI')
            if not os.path.exists(lmci_path):
                os.makedirs(lmci_path)
            dest_path = os.path.join(lmci_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))
        elif df[df['Subject'] == folder]['Group'].values[0] == 'EMCI':
            emci_path = os.path.join(out_path, 'EMCI')
            if not os.path.exists(emci_path):
                os.makedirs(emci_path)
            dest_path = os.path.join(emci_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))
        elif df[df['Subject'] == folder]['Group'].values[0] == 'SMC':
            smc_path = os.path.join(out_path, 'SMC')
            if not os.path.exists(smc_path):
                os.makedirs(smc_path)
            dest_path = os.path.join(smc_path, folder)
            # 查找原文件夹下面的localizer子文件夹
            localizer_path = os.path.join(folder_path, 'localizer')
            # 查看localizer文件夹是否存在
            if not os.path.exists(localizer_path):
                continue
            # 将localizer子文件夹复制到dest_path下
            shutil.copytree(localizer_path, os.path.join(dest_path, 'localizer'))
import os
import shutil
import numpy as np

os.chdir('D:/vscodes/AIinMed/final_project/AD_classification_python')

## 数据位置
train_dir = './netDataset/training'
val_dir = './netDataset/validation'
test_dir = './netDataset/testing'

## 获取每个类别的文件夹
train_class = os.listdir(train_dir)
val_class = os.listdir(val_dir)
test_class = os.listdir(test_dir)

## 统计每个类别的数量
train_class_num = []
val_class_num = []
test_class_num = []
for i in range(len(train_class)):
    train_class_num.append(len(os.listdir(train_dir + '/' + train_class[i])))
    val_class_num.append(len(os.listdir(val_dir + '/' + val_class[i])))
    test_class_num.append(len(os.listdir(test_dir + '/' + test_class[i])))

## 绘制柱状图
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.bar(train_class, train_class_num, label='train')
plt.bar(val_class, val_class_num, label='val/test')
plt.legend()
plt.show()

## 统计一下总数
print('train:', np.sum(train_class_num))
print('val:', np.sum(val_class_num))
print('test:', np.sum(test_class_num))
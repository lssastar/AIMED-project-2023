import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pretrainedmodels

# %matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5  # Adjust based on your classes
labels = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']

# 导入保存的模型
## 1. Alexnet
alexnet = models.alexnet(pretrained=True)
# Replace the classifier's last fully connected layer
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, NUM_CLASSES)

alexnet.load_state_dict(torch.load('./models/alexnet_400.pth'))

## 2. Resnet50
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, NUM_CLASSES)  # 修改全连接层

# 加载状态字典或完整模型
loaded_data = torch.load('./models/Resnet50_200.pth')

if isinstance(loaded_data, dict):
    # 如果是状态字典
    resnet50.load_state_dict(loaded_data)
elif isinstance(loaded_data, models.ResNet):
    # 如果是完整模型
    resnet50 = loaded_data
else:
    raise TypeError("加载的数据既不是 state_dict 也不是 ResNet 模型。")

## 3. Resnet101
resnet101 = models.resnet101(pretrained=True)
resnet101.fc = nn.Linear(resnet101.fc.in_features, NUM_CLASSES)  # 修改全连接层

# 加载状态字典或完整模型
loaded_data = torch.load('./models/Resnet101_200.pth')

if isinstance(loaded_data, dict):
    # 如果是状态字典
    resnet101.load_state_dict(loaded_data)
elif isinstance(loaded_data, models.ResNet):
    # 如果是完整模型
    resnet101 = loaded_data
else:
    raise TypeError("加载的数据既不是 state_dict 也不是 ResNet 模型。")

## 4. InceptionResnetV2
inceptionresnetv2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=None)

pretrained_dict = torch.load('./models/inceptionresnetv2-520b38e4.pth')
model_dict = inceptionresnetv2.state_dict()

# Remove weights for last_linear layer from pretrained_dict
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['last_linear.weight', 'last_linear.bias']}

# Update the current model's state_dict
model_dict.update(pretrained_dict)

inceptionresnetv2.load_state_dict(model_dict)

inceptionresnetv2.last_linear = nn.Linear(inceptionresnetv2.last_linear.in_features, NUM_CLASSES)  # 修改全连接层

inceptionresnetv2 = torch.load('./models/inceptionresnetv2_150.pth')

## 5. GoogleNet
googlenet = models.googlenet(pretrained=True)
googlenet.fc = nn.Linear(googlenet.fc.in_features, NUM_CLASSES)  # 修改全连接层

googlenet = torch.load('./models/Googlenet_400.pth')


##把模型放到GPU上
alexnet.to(device)
resnet50.to(device)
resnet101.to(device)
inceptionresnetv2.to(device)
googlenet.to(device)


# 导入测试集: resnet50, resnet101, GoogleNet
val_dir = './netDataset/validation'
test_transforms_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

val_dataset224 = datasets.ImageFolder(val_dir, test_transforms_224)
val_loader224 = DataLoader(val_dataset224, batch_size=1, shuffle=False)

# 导入测试集, alexnet
val_dir = './netDataset/validation'
test_transforms_227 = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

val_dataset227 = datasets.ImageFolder(val_dir, test_transforms_227)
val_loader227 = DataLoader(val_dataset227, batch_size=1, shuffle=False)

# 导入测试集, inceptionresnetv2
val_dir = './netDataset/validation'
test_transforms_299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

val_dataset299 = datasets.ImageFolder(val_dir, test_transforms_299)
val_loader299 = DataLoader(val_dataset299, batch_size=1, shuffle=False)

## 集成预测
# 1. 集成预测函数
def ensemble(net1, net2, net3, net4, net5, val_loader1, val_loader2, val_loader3, val_loader4, val_loader5):
    """
    net1: alexnet
    net2: resnet50
    net3: resnet101
    net4: inceptionresnetv2
    net5: googlenet
    """
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for data1, data2, data3, data4, data5 in tqdm(zip(val_loader1, val_loader2, val_loader3, val_loader4, val_loader5)):
            images1, labels1 = data1
            images2, labels2 = data2
            images3, labels3 = data3
            images4, labels4 = data4
            images5, labels5 = data5

            # 将数据送到相应设备
            images1, labels1 = images1.to(device), labels1.to(device)
            images2 = images2.to(device)
            images3 = images3.to(device)
            images4 = images4.to(device)
            images5 = images5.to(device)

            # 计算每个模型的输出
            outputs1 = net1(images1)
            outputs2 = net2(images2)
            outputs3 = net3(images3)
            outputs4 = net4(images4)
            outputs5 = net5(images5)

            # 获取每个模型的最大概率的索引（即预测）
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            _, preds3 = torch.max(outputs3, 1)
            _, preds4 = torch.max(outputs4, 1)
            _, preds5 = torch.max(outputs5, 1)

            # 汇总所有模型的预测并进行投票
            final_preds = []
            for i in range(preds1.size(0)):
                votes = [preds1[i], preds2[i], preds3[i], preds4[i], preds5[i]]
                vote_counts = torch.bincount(torch.stack(votes))
                final_pred = torch.argmax(vote_counts)
                final_preds.append(final_pred)

            final_preds = torch.stack(final_preds)

            # 更新正确预测数和总数
            total += labels1.size(0)
            correct += (final_preds == labels1).sum().item()

            # 存储预测结果
            predictions.append(final_preds.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
    return predictions

# 评估集成模型
pred = ensemble(alexnet, resnet50, resnet101, inceptionresnetv2, googlenet, val_loader227, val_loader224, val_loader224, val_loader299, val_loader224)

# 2. 评估集成模型
# 2.1 混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools

y_true = val_dataset227.targets
y_pred = np.concatenate(pred)
cm = confusion_matrix(np.array(y_true), y_pred)
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    此函数打印并绘制混淆矩阵。
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']
plt.figure()
plot_confusion_matrix(cm, labels, title='Confusion matrix')
# plt.show()
plt.savefig('./docs/ensembleSumVote_confusion_matrix.png')

# 2.2 ROC曲线
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 计算每个样本的得分
y_true = []
y_pred = []  # 存储最终的预测结果
alexnet.eval()
resnet50.eval()
resnet101.eval()
inceptionresnetv2.eval()
googlenet.eval()

with torch.no_grad():
    for data1, data2, data3, data4, data5 in tqdm(zip(val_loader227, val_loader224, val_loader224, val_loader299, val_loader224)):
        images1, labels1 = data1
        images2, labels2 = data2
        images3, labels3 = data3
        images4, labels4 = data4
        images5, labels5 = data5

        # 将图像数据送到设备
        images1 = images1.to(device)
        images2 = images2.to(device)
        images3 = images3.to(device)
        images4 = images4.to(device)
        images5 = images5.to(device)

        # 获取每个模型的输出
        outputs1 = alexnet(images1)
        outputs2 = resnet50(images2)
        outputs3 = resnet101(images3)
        outputs4 = inceptionresnetv2(images4)
        outputs5 = googlenet(images5)

        # 获取每个模型输出的最大概率的索引（即预测类别）
        _, preds1 = torch.max(outputs1, 1)
        _, preds2 = torch.max(outputs2, 1)
        _, preds3 = torch.max(outputs3, 1)
        _, preds4 = torch.max(outputs4, 1)
        _, preds5 = torch.max(outputs5, 1)

        # 汇总所有模型的预测并进行投票
        final_preds = []
        for i in range(preds1.size(0)):
            votes = [preds1[i], preds2[i], preds3[i], preds4[i], preds5[i]]
            vote_counts = torch.bincount(torch.stack(votes))
            final_pred = torch.argmax(vote_counts)
            final_preds.append(final_pred)

        final_preds = torch.stack(final_preds)

        # 存储最终的预测和真实标签
        y_pred.append(final_preds.cpu().numpy())
        y_true.append(labels1.cpu().numpy())

# 在循环外处理结果
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

## 设置颜色循环
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
labels = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']

# Plot the ROC curve for each class
plt.figure(figsize=(7, 7))
for i, color, lbl in zip(range(NUM_CLASSES), colors, labels):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(lbl, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('./docs/ensembleSumVote_ROC.png')

# 2.3 模型评估指标
from sklearn.metrics import classification_report
y_true = val_dataset227.targets
y_pred = np.concatenate(pred)
print(classification_report(np.array(y_true), y_pred, target_names=np.array(labels)))
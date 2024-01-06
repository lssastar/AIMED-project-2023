######################### Use CNN to classify the images #########################
######################### 使用ResNet101进行迁移学习 #########################
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

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 5  # Adjust based on your classes
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 0.0001
val_freq = 10
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('D:/vscodes/AIinMed/final_project/AD_classification_python')

# 保存模型位置
save_dir = './models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Define your dataset
data_dir = './netDataset'
train_dir = data_dir + '/training'
val_dir = data_dir + '/validation'
test_dir = data_dir + '/testing'

# Apply transformations (normalization, augmentation, etc.)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_val_transforms = transforms.Compose([
    transforms.RandomRotation((-35, 35)),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.5, 1.0)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# load the dataset
train_dataset = datasets.ImageFolder(train_dir, train_val_transforms)
val_dataset = datasets.ImageFolder(val_dir, test_transforms)
test_dataset = datasets.ImageFolder(test_dir, test_transforms)

# create the dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load pre-trained ResNet101
net = models.resnet101(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)  # 修改全连接层

# Move the network to the GPU if available
net = net.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Train the network
train_loss = []
val_loss = []
train_acc = []
val_acc = []

for epoch in range(EPOCHS):
    ############################
    # Train
    ############################
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # 使用tqdm包裹您的训练加载器
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit='batch')):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    ############################
    # Validate
    ############################
    if epoch % val_freq == 0:
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)
        net.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        # 使用tqdm包裹您的验证加载器
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validate]", unit='batch')):
                # Get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Print statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss.append(running_loss / len(val_loader))
        val_acc.append(correct / total)

        print('[%d] Train loss: %.3f' %
              (epoch + 1, train_loss[-1]))
        print('[%d] Val loss: %.3f' %
              (epoch + 1, val_loss[-1]))
        print('[%d] Train acc: %.3f' %
              (epoch + 1, train_acc[-1]))
        print('[%d] Val acc: %.3f' %
              (epoch + 1, val_acc[-1]))
        
# Plot the loss and accuracy curves
plt.figure()
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('./docs/Resnet101_200_loss.png') ### 每次修改模型时，需要修改这里的文件名

plt.figure()
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./docs/Resnet101_200_acc.png') ### 每次修改模型时，需要修改这里的文件名

# 保存模型
torch.save(net, save_dir + '/Resnet101_200.pth') ### 每次修改模型时，需要修改这里的文件名

# 在测试集上测试
## 混淆矩阵
from sklearn.metrics import confusion_matrix
import itertools

y_true = []
y_pred = []
net.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, desc=f"Testing", unit='batch')):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

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
plt.savefig('./docs/Resnet101_200_confusion_matrix.png') ### 每次修改模型时，需要修改这里的文件名

## ROC曲线
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 计算每个样本的得分
net.eval()
y_score = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, desc=f"Testing", unit='batch')):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = net(inputs)
        y_score.append(outputs.cpu().numpy())
        y_true.append(labels.cpu().numpy())

y_score = np.concatenate(y_score)
y_true = np.concatenate(y_true)

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
plt.savefig('./docs/Resnet101_200_ROC.png') ### 每次修改模型时，需要修改这里的文件名
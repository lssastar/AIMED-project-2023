######################### Use CNN to classify the images #########################
######################### 使用GoogleNet进行迁移学习 #########################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import numpy as np
# %matplotlib inline

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 5  # Adjust based on your classes
BATCH_SIZE = 128
EPOCHS = 400
LEARNING_RATE = 0.0001
val_freq = 10
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('D:/vscodes/AIinMed/final_project/AD_classification_python')

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

# Load pre-trained GoogLeNet
net = models.googlenet(pretrained=True)
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
    for i, data in enumerate(train_loader):
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
        with torch.no_grad():
            for i, data in enumerate(val_loader):
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
plt.savefig('./netDataset/Googlenet_400_loss.png')

plt.figure()
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./netDataset/Googlenet_400_acc.png')

# 保存模型
torch.save(net, './netDataset/Googlenet_400.pth')

## 混淆矩阵
from sklearn.metrics import confusion_matrix
import itertools
net.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.cpu().numpy())
        y_true.append(labels.cpu().numpy())

cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred))

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
plt.savefig('./netDataset/Googlenet_400_cm.png')

## ROC曲线
# Plot the ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Calculate the score for each sample
net.eval()
y_score = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
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

# Calculate the ROC curve and AUC for each class
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
plt.savefig('./netDataset/Googlenet_400_ROC.png')
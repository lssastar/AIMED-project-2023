######################### Use CNN to classify the images #########################
######################### 使用Alexnet进行迁移学习 #########################
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
IMG_WIDTH, IMG_HEIGHT = 227, 227
NUM_CLASSES = 5  # Adjust based on your classes
BATCH_SIZE = 128
EPOCHS = 400
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

# Load pre-trained AlexNet
net = models.alexnet(pretrained=True)

# Replace the classifier's last fully connected layer
net.classifier[6] = nn.Linear(net.classifier[6].in_features, NUM_CLASSES)

# Replace the last layer with a log softmax layer for classification
net.classifier.add_module("7", nn.LogSoftmax(dim=1))

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
plt.show()

plt.figure()
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
torch.save(net.state_dict(), save_dir + '/alexnet_400.pth')

# Evaluate the model
net.eval()
total = 0
correct = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = net(inputs)

        # Print statistics
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %.3f %%' % (total, 100 * correct / total))

# 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 计算每个样本的得分
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

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有类别平均的ROC曲线
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='Average ROC curve (area = %0.2f)' % roc_auc["micro"])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

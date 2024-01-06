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

os.chdir('D:/vscodes/AIinMed/final_project/AD_classification_python')
data_dir = './netDataset'

NUM_CLASSES = 5  # Adjust based on your classes
labels = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']

IMG_WIDTH, IMG_HEIGHT = 227, 227

# 导入保存的模型
## 1. Alexnet
alexnet = models.alexnet(pretrained=True)
# Replace the classifier's last fully connected layer
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, NUM_CLASSES)

alexnet.load_state_dict(torch.load('./models/alexnet_400.pth'))

alexnet.to(device)

test_dir = data_dir + '/testing'

# Apply transformations (normalization, augmentation, etc.)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Test the model
## 混淆矩阵
from sklearn.metrics import confusion_matrix
import itertools
alexnet.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = alexnet(inputs)
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
plt.savefig('./docs/alexnet_200_confusion_matrix.png') ### 每次修改模型时，需要修改这里的文件名

## ROC曲线
# Plot the ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Calculate the score for each sample
alexnet.eval()
y_score = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        outputs = alexnet(inputs)
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
plt.savefig('./docs/alexnet_200_ROC.png') ### 每次修改模型时，需要修改这里的文件名
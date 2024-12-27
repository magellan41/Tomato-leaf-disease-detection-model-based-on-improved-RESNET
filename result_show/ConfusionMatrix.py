import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from sklearn.model_selection import train_test_split
from torchvision.models import ResNet34_Weights

from cnn.ResNetWIthSe import ResNetWithSE

from cnn.ResNetWithCBAM import ResNetWithCBAM

if __name__ == '__main__':

    diseases = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
                'Spider_mites Two-spotted_spider_mite',
                'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy']

    model_name_list = [# 'AlexNet',
                       # 'AlexNetModified',
                       'ResNet',
                       'ResNet_pretrained',
                       'ResNet_CBAM',
                       'ResNet_SELayer']
    model_list = [# AlexNet(),
                  # AlexNetModified(),
                  models.resnet34(weights=ResNet34_Weights.DEFAULT),
                  models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
                  ResNetWithCBAM(models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)),
                  ResNetWithSE(models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义数据预处理变换
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 数据集路径
    data_dir = '../data/PlantVillage'

    # 创建 ImageFolder 数据集对象
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

    # 获取索引列表
    indices = list(range(len(dataset)))

    # 使用 train_test_split 来划分训练集和验证集
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=dataset.targets, random_state=42)

    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 打印一些信息
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images across {len(dataset.classes)} classes.")

    # 计算准确率、召回率、F1 值
    for model_name, m in zip(model_name_list, model_list):
        dic_path = f'../model/{model_name}_final.pth'

        # 实例化模型
        model = m.to(device)
        model.load_state_dict(torch.load(dic_path, weights_only=True))
        model.eval()

        # 生成混淆矩阵
        confusion_matrix = np.zeros((len(dataset.classes), len(dataset.classes)))
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                if pred > len(dataset.classes) - 1:
                    continue
                confusion_matrix[label, pred] += 1

        plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
        plt.xticks(np.arange(len(dataset.classes)), diseases, rotation=45)
        plt.yticks(np.arange(len(dataset.classes)), diseases)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.legend()
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()



        # 计算准确率、召回率、F1 值
        for i in range(len(dataset.classes)):
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP
            TN = np.sum(confusion_matrix) - TP - FP - FN
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            print(f'{model_name} {dataset.classes[i]} Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
        print()

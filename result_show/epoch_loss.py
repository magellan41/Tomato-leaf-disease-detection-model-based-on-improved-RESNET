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

    accuracies_list = []

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

    epoch_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for model_name, m in zip(model_name_list, model_list):

        # 实例化模型
        model = m.to(device)
        accuracy_list = []
        for epoch in epoch_list:
            dic_path = f'../model/{model_name}_{epoch}.pth'
            model.load_state_dict(torch.load(dic_path, weights_only=True))
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            accuracy_list.append(accuracy)
        accuracies_list.append(accuracy_list)

    plt.figure(figsize=(10, 10))
    for model_name, accuracies in zip(model_name_list, accuracies_list):
        plt.plot(epoch_list, accuracies, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()







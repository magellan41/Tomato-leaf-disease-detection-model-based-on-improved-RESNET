import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models

from sklearn.model_selection import train_test_split
from torchvision.models import ResNet34_Weights

from cnn.ResNetWIthSe import ResNetWithSE

from cnn.ResNetWithCBAM import ResNetWithCBAM


def evaluate_model(model, model_name, val_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of {model_name} on the {total} validation images: {100 * correct / total}%")



def train_model(model,model_name, train_loader, optimizer, criterion, device, epochs=10):

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 9:
            evaluate_model(model, model_name, val_loader, device)
            torch.save(model.state_dict(), f'model/{model_name}_{epoch + 1}.pth')



if __name__ == '__main__':
    model_name_list = [# 'AlexNet',
                       # 'AlexNetModified',
                       # 'ResNet',
                       # 'ResNet_pretrained',
                       'ResNet_CBAM_SGD',
                       'ResNet_SELayer_SGD']
    model_list = [# AlexNet(),
                  # AlexNetModified(),
                  # models.resnet34(weights=ResNet34_Weights.DEFAULT),
                  # models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
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
    data_dir = './data/PlantVillage'

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

    for model_name, m in zip(model_name_list, model_list):
        dic_path = f'model/{model_name}_final.pth'

        # 实例化模型
        model = m.to(device)

        # 训练模型或者加载已经训练完成的模型
        if not os.path.exists(dic_path):
            # 定义损失函数和优化器
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            train_model(model, model_name, train_loader, optimizer, criterion, device, epochs=100)
            if not os.path.exists('model'):
                os.mkdir('model')
            torch.save(model.state_dict(), dic_path)
        else:
            model.load_state_dict(torch.load(dic_path, weights_only=True))

        # 评估模型
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
            print(f"Accuracy of {model_name} on the {total} validation images: {100 * correct / total}%")


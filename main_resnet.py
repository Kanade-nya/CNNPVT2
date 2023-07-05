# 设置全局参数
import json

import torch
import torchvision.datasets
from torch import nn, optim
from torchvision import transforms
from pvt import pvt_tiny
from torch.autograd import Variable
from torchvision.models import resnet50

from resnet_50 import ResNet50

modellr = 1e-4
BATCH_SIZE = 128
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_ds = torchvision.datasets.CIFAR100('data',
                                         train=True,
                                         transform=transform_test,  # 将数据类型转化为Tensor
                                         download=True)

test_ds = torchvision.datasets.CIFAR100('data',
                                       train=False,
                                       transform=transform_test,  # 将数据类型转化为Tensor
                                       download=True)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
model_ft = ResNet50(num_classes=100)
# print(model_ft)

# num_ftrs = model_ft.head.in_features
# model_ft.head = nn.Linear(num_ftrs, 12,bias=True)
# nn.init.xavier_uniform_(model_ft.head.weight)
model_ft.to(DEVICE)
# print(model_ft)
# # 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)


alpha=0.2
def train(model, device, train_loader, optimizer, epoch,log_stats):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)

    correct = 0
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # data, labels_a, labels_b, lam = mixup_data(data, target, alpha)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        # loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        # 计算
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)

        print_loss = loss.data.item()
        sum_loss += print_loss

        if (batch_idx + 1) % 10 == 0:
            train_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(),lr)
            print(train_str)

    ave_loss = sum_loss / len(train_loader)

    correct = correct.data.item()
    acc = correct / total_num

    print('\nTrain set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset), 100 * acc))


    print('epoch:{},loss:{}'.format(epoch, ave_loss))

    str1 = '[Train set:  Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(train_loader.dataset), 100 * acc) +  ', epoch:{},loss:{},'.format(epoch, ave_loss)

    log_stats.append(str1)
ACC=0
# 验证过程
def val(model, device, test_loader):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            # output = model(data)
            pred = model(data)
            # loss = criterion(pred, target).item()

            test_loss += criterion(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            # _, pred = torch.max(output.data, 1)
            # correct += torch.sum(pred == target)
            # # print_loss = loss.data.item()
            # test_loss += print_loss
        # correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        test_str = ' Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ]'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        log_stats.append(str(test_str))
        torch.save(model_ft,'modelresnet.pth')
        if acc > ACC:
            # torch.save(model_ft, 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            ACC = acc


# 训练

for epoch in range(1, EPOCHS + 1):
    log_stats = []
    train(model_ft, DEVICE, train_loader, optimizer, epoch,log_stats)
    cosine_schedule.step()
    val(model_ft, DEVICE, test_loader)
    with open('output_dir/logresnet.txt', 'a') as f:
        f.write(json.dumps(log_stats) + "\n")
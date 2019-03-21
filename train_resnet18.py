from datasetw import DataSetW
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pandas as pd
import time
from torch.nn.init import xavier_uniform_ as xavier
from torch.nn.init import kaiming_uniform_ as kaiming
import json

def init_weights(m, method=xavier):
    """
    初始化卷积层的权值，默认使用xavier_uniform
    :param m: 需要进行初始化的模块
    :param method: 初始化方式
    """
    if isinstance(m, nn.Conv2d):
        method(m.weight.data)

def init_weights2(m, method=kaiming):
    if isinstance(m, nn.Conv2d):
        method(m.weight.data)


def main():
    train_conf = json.load(open("cong/train.json"))
    # 测试集占的比重
    test_size = train_conf['test_size']

    # 对图片进行像素归一化处理（0-1）
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

    img_df = pd.read_csv(train_conf['label_path'])
    # shuffle dataset
    img_df = img_df.sample(frac=1)
    split = int(len(img_df) * test_size)

    trainset = DataSetW(img_df[split:].reset_index(drop=True),
                        img_path='data/processed_img',
                        transformer=transformer)
    testset = DataSetW(img_df[0:split].reset_index(drop=True),
                       img_path='data/processed_img',
                       transformer=transformer)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128)

    # cuda support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 选择torch内置的resnet18模型
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Linear(512, train_conf['class_count']),
                             nn.LogSoftmax(dim=1))

    # 对模型进行初始化
    model.apply(init_weights)

    # 使用多个GPU并行计算
    model = nn.DataParallel(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # 记录训练过程中的损失及准确率
    losses = []
    testlosses = []
    accs = []

    epochs = train_conf['epochs']
    t_start = time.time()
    best_accuracy = 0
    map_epoch = 0
    model.train()
    for epoch in range(0, epochs + 1):
        running_loss = 0
        print('Training: epoch [%d / %d]' % (epoch, epochs))
        for imgs, clazzs in trainloader:
            imgs, clazzs = imgs.to(device), clazzs.to(device)
            optimizer.zero_grad()
            losps = model.forward(imgs)
            loss = criterion(losps, clazzs)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 记录每个epoch的训练损失
        losses.append(running_loss)

        # # 第30个epoch的时候对优化器的学习率进行调整
        # if epoch == 30:
        #     for paramgroup in optimizer.param_groups:
        #         paramgroup['lr'] = 0.0003

        # 每5个epoch对网络的识别率进行检测
        if epoch % 2 == 0:
            with torch.no_grad():
                # 修改模型为评估模式
                model.eval()
                accuracy = 0
                testloss = 0
                for imgs, clazzs in testloader:
                    imgs, clazzs = imgs.to(device), clazzs.to(device)
                    output = model.forward(imgs)
                    testloss += criterion(output, clazzs).item()
                    output = torch.exp(output)
                    _, top_class = output.topk(1, dim=1)
                    equals = top_class == clazzs.view(*top_class.shape)
                    accuracy += torch.sum(equals.float()).item()
                model.train()
                testlosses.append(testloss)
                accuracy = accuracy / len(testset)
                accs.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    map_epoch = epoch
    t_end = time.time()
    torch.save(model.state_dict(), 'resnet18.pth')

    # 绘制训练损失，测试损失，测试准确率
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(losses, color='#4169E1', label='Running loss')
    axs[1].plot(testlosses, color='#FFA500', label='Test loss')
    axs[2].plot(accs, color='#00FF7F', label='Accuracy')
    print("Training cost %.1f seconds!" % (t_end - t_start))
    print("Best accuracy: %.3f" % best_accuracy)
    print("Mapped epoch: %d" % map_epoch)
    fig.legend()
    fig.savefig('result18h.png')


if __name__ == '__main__':
    main()

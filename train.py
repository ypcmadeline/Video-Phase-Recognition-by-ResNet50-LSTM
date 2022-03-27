import random
import shutil
import time
import warnings
# from thop import profile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import torch.nn.init as init
from dataset import Surgery
from matplotlib import pyplot as plt

train_transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(60),
    ])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs =10
sequence_length = 3
learning_rate = 5e-4
loss_layer = nn.CrossEntropyLoss().to(device)

loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []

def plot(loss_train, loss_test, acc_train, acc_test):
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(loss_train, label='Train')
    axis[0].plot(loss_test, label='Valid')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('loss')
    axis[0].set_title("Training and Validation Loss")
    axis[1].plot(acc_train, label='Train')
    axis[1].plot(acc_test, label='Valid')
    axis[1].legend(['Train','Valid'])
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('Accuracy')
    axis[1].set_title("Training and Validation Accuracy")
    plt.show()
    plt.savefig("graph.jpg")

class ResNetLSTM(nn.Module):
    def __init__(self):
        super(ResNetLSTM, self).__init__()
        print("LSTM")
        self.resnet = torchvision.models.resnet50(pretrained=True)       
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 2048))
        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        self.LSTM=torch.nn.LSTM(2048, 512, 3, batch_first=True)
        self.fc=torch.nn.Linear(512,7)
    
    def forward(self, x):
        batch_size = x.size(0)
        sequence = x.size(1)
        img = torch.Tensor(batch_size,sequence,2048).cuda()
        for s in range(sequence_length):
            img[:,s,:] = self.resnet(x[:,s,:,:,:])

        seq ,_ = self.LSTM(img)

        seq = seq[:,-1,:]

        out = self.fc(seq)

        return out


def train(model,train_loader, test_loader, learning_rate):  
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    # print(learning_rate)
    for epoch in range(1, epochs + 1):
        if epoch % 2 == 0:
            learning_rate = learning_rate * 0.5
            optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        model.train()
        
        correct = 0
        total = len(train_dataloader.dataset)
        loss_item = 0

        for data in tqdm(train_loader):
            ## your code
            optimizer.zero_grad()

            images, target = data

            if use_cuda:
                images = images.cuda()
                target = target.cuda()

            outputs = model(images)
                
            loss = loss_layer(outputs, target)
            # loss = Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            loss_item += loss.item()
            correct += torch.sum(pred == target)
        
        loss = loss_item / total
        loss_train_list.append(loss)
        acc_train = correct / total
        acc_train_list.append(acc_train.item())

        print('Train: Acc {}, Loss {}'.format(acc_train, loss))

        test(model, test_loader)
        
        

           
def test(model,test_loader):
    print('Testing...')
    model.eval()
    correct = 0
    total = len(test_dataloader.dataset)
    loss_item = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, target = data

            if use_cuda:
                images = images.cuda()
                target = target.cuda()

            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            loss = loss_layer(outputs, target)
            loss_item += loss.item()
            correct += torch.sum(pred == target)
        
        loss = loss_item / total
        loss_test_list.append(loss)
        acc_test = correct / total
        acc_test_list.append(acc_test.item())


    print('Test: Acc {}, Loss {}'.format(acc_test, loss))
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':

    part = 1

    if part == 2:
        print("Running ResNet+LSTM")
        model = ResNetLSTM()
    else:
        print("Running ResNet50")
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Sequential(nn.Linear(2048, 7))   
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

    traindataset = Surgery(transform=train_transform)
    train_dataloader = DataLoader(traindataset, batch_size=32, shuffle=True, drop_last=True)
    testdataset = Surgery(train=False, transform=test_transform)
    test_dataloader = DataLoader(testdataset, batch_size=32, shuffle=False, drop_last=False)
    train(model, train_dataloader,test_dataloader,learning_rate)

    plot(loss_train_list, loss_test_list, acc_train_list, acc_test_list)
   
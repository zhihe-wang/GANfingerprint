import os
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

#%% Model & data
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.avgpool = nn.AvgPool2d(2,2)
        
    def forward(self, img):
        out = self.conv1(img)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        return out

class AttrNet(nn.Module):
    def __init__(self):
        super(AttrNet,self).__init__()
        self.layer1 = Block(3, 16)
        self.layer2 = Block(16, 32)
        self.layer3 = Block(32, 64)
        self.layer4 = Block(64, 128)
        self.layer5 = Block(128, 256)
        self.adapool = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(1024, 6)
    
    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.adapool(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out 

'''
|-- Dataset2
    |-- Training
        |-- Original
        |-- Deepfakes
        |-- DeepFakeDetection
        |-- Face2Face
        |-- FaceSwap
        |-- NeuralTextures
    |-- Validation
        |-- Original
        ...
'''

# Load model
model = AttrNet().cuda()
summary(model, input_size=(3, 224, 224))

#%% Train
# Hyper-parameters
BATCH_SIZE = 16
EPOCH = 5
LR = 1e-4
train_path =  '../Dataset2/Training'
val_path = '../Dataset2/Validation'

# Load data
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_path, transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = datasets.ImageFolder(val_path, transform)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

# Training method
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

model.load_state_dict(torch.load('./attrNet.pth'))
for epoch in range(EPOCH):
    # Training
    model.train()
    run_loss = []
    for step, (image, label) in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output,label)
        loss.backward()
        optimizer.step()
        
        run_loss.append(loss.item())        
        if step % 100 == 0 and step != 0:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            string = 'Epoch:{}, Step:{}, Training Loss:{:.2f}'.format(
                epoch, step, sum(run_loss)/len(run_loss))
            print(string)
            with open('log.txt','a') as log:
                print(string, file=log)
            run_loss = []
    lr_scheduler.step()
    torch.save(model.state_dict(), './attrNet1.pth')
        
    # Validation
    model.eval()
    y_true, y_pred = [], []
    for step, (image, label) in enumerate(val_loader):
        image = image.cuda()
        
        y_pred.extend(model(image).tolist())
        y_true.extend(label.tolist())
    
    y_true, y_pred = np.array(y_true), np.argmax(np.array(y_pred), axis=1)
        
    acc = accuracy_score(y_true, y_pred)
    string = 'Epoch:{}, Validation Accuracy:{:.2%}.'.format(epoch, acc)
    print(string)
    with open('log.txt','a') as log:
        print(string, file=log)

#%%




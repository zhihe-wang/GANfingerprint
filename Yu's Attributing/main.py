import os
import time

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
        self.relu = nn.LeakyReLU()
        self.avgpool = nn.AvgPool2d(2,2)
        
    def forward(self, img):
        out = self.conv1(img)
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
        self.conv1 = nn.Conv2d(256, 512, 7)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(512, 6)
    
    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.relu(self.conv1(out))
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

#%% Train
# Load model
model = AttrNet().cuda()
summary(model, input_size=(3, 224, 224))
# model_path = './attrNet1.pth'
# model.load_state_dict(torch.load(model_path))

# Hyper-parameters
BATCH_SIZE = 4
EPOCH = 3
LR = 0.000000001
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
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

for epoch in range(EPOCH):
    # Training
    model.train()
    all_loss = []
    for step, (image, label) in enumerate(train_loader):
        time_start=time.time()
        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output,label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        all_loss.append(loss.item())
        
        time_end=time.time()
        if step % 100 == 0 and step != 0:
            print('Epoch:{}, Step:{}, Training Loss:{:.2f}, Step Time:{:.2f}s'.
                  format(epoch, step, sum(all_loss)/100, time_end-time_start))
            all_loss = []
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
    print('Epoch:{}, Validation Accuracy:{:.2%}.'.format(epoch, acc))





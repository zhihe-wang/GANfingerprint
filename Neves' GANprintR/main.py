import os
import sys
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchsummary import summary
from sklearn.metrics import accuracy_score, recall_score

from xceptionnet import xception

#%% Model
# model = xception(num_classes=1000, pretrained='imagenet')
# model.num_classes = 1
# dim_feats = model.fc.in_features  # =2048
# model.fc = nn.Linear(dim_feats, 1)
model = xception(num_classes=1)
model.cuda()
summary(model, input_size=(3,224,224))

#%% Train
BATCH_SIZE = 16
EPOCH = 5
LR = 1e-4

train_path = "./Dataset/Train_DF/"
val_path = "./Dataset/Val_DF/"
train_trans = transforms.Compose([transforms.ColorJitter(0.01,0.01,0.01,0.01),
                                  transforms.ToTensor(),
                                  transforms.RandomErasing(0.3,(0.02,0.1),(1,1))])

train_dataset = datasets.ImageFolder(train_path, train_trans)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True)
val_dataset = datasets.ImageFolder(val_path, transforms.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model.load_state_dict(torch.load('./XceptionNet/step_last.pth'))
for epoch in range(EPOCH):
    # Training
    model.train()
    run_loss = []
    for step, (inputs, labels) in enumerate(train_loader):
        if step == 0:
            save_image(inputs, './XceptionNet/img/train.jpg', nrow=4)
        inputs = inputs.cuda()
        labels = labels.cuda().type(torch.float)
        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.reshape(outputs.size())
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        run_loss.append(loss.item())
        # print statistics
        if step%20 == 0 and step != 0:
           print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
           print('Epoch:{}|Step:{}, Training loss: {:.3f}.\n'
                 .format(epoch, step, sum(run_loss)/len(run_loss)))
           run_loss = []
    torch.save(model.state_dict(),'./XceptionNet/step_last1.pth')
    lr_scheduler.step()
    
    # Validation
    model.eval()
    y_true, y_pred = [], []
    for step, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.cuda()
        
        y_pred.extend(model(inputs).sigmoid().tolist())
        y_true.extend(labels.tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)>0.5

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print('Epoch:{}, Validation Accuracy:{:.2%}, Recall:{:.2%}.\n'
          .format(epoch, acc, rec))
    
#%% Test
def GaussianBlur(radius):
    l = lambda x:x.filter(ImageFilter.GaussianBlur(radius))
    return transforms.Lambda(l)

BATCH_SIZE = 16
# test: (data_path,data_transform)
data_path = {
'DF_original':"./Dataset/Val_DF",
'DF_printR':"./Dataset/Val_DF_printR",
'DF_gaussian':"./Dataset/Val_DF",
'F2F_original':"./Dataset/Val_F2F",
'NT_original':"./Dataset/Val_NT"}

data_trans = {
'DF_original':transforms.ToTensor(),
'DF_printR':transforms.ToTensor(),
'DF_gaussian':transforms.Compose([GaussianBlur(1.2),transforms.ToTensor()]),
'F2F_original':transforms.ToTensor(),
'NT_original':transforms.ToTensor()}

model.load_state_dict(torch.load('./XceptionNet/step_last.pth'))
model.eval()
for test in data_path.keys():
    dataset = datasets.ImageFolder(data_path[test], data_trans[test])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []
    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.cuda()
        if step == 0:
            save_image(inputs, './XceptionNet/img/'+test+'.jpg', nrow=4)
        
        y_pred.extend(model(inputs).sigmoid().tolist())
        y_true.extend(labels.tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)>0.5
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print('Test:{}, Test Accuracy:{:.2%}, Test Recall:{:.2%}.\n'
          .format(test, acc, rec))
# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from resnet import resnet50

#%% Data process

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#%% Model
model_path = './blur_jpg_prob0.5.pth'
# model_path = './fine_tuned.pth'

model = resnet50(num_classes=1)
model.load_state_dict(torch.load('./blur_jpg_prob0.5.pth')['model'])
# model.load_state_dict(torch.load('./fine_tuned.pth'))
model.cuda()

#%% Train
dataset_path =  '../Dataset/'
# dataset_path = r'G:\Repos\CNNDetection\dataset\test\gaugan'

train_data = datasets.ImageFolder(dataset_path, transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
model.train()

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

for idx, param in enumerate(model.parameters()):
    if idx == 160: break
    param.requires_grad = False

for epoch in range(100):
    for step, (image, label) in enumerate(train_loader):
        time_start=time.time()
        image = image.cuda()
        label = label.cuda().float()

        optimizer.zero_grad()
        output = model(image).flatten()
        loss = loss_function(output,label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        time_end=time.time() 
        if step%20 == 0:
            print('Epoch:{}, Step:{}, Loss:{:.2f}, Time:{:.2f}s'.
                  format(epoch, step, loss.item(), time_end-time_start))
            
            # print('',output,'\n\n',label,'\n\n',loss,'\n\n')
        
torch.save(model.state_dict(), './fine_tuned_2.pth')

#%% Test
dataset_path =  '../Dataset/NeuralTextures'
# dataset_path = r'G:\Repos\CNNDetection\dataset\test\deepfake'

test_data = datasets.ImageFolder(dataset_path, transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
model.eval()

time_start=time.time()

with torch.no_grad():
    y_true, y_pred = [], []
    for image, label in test_loader:
        image = image.cuda()
        
        y_pred.extend(model(image).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    threshold = 0.000001
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)
    
    print(('Model: {}\nTest on: {}\nReal accuracy:{:.2%}, Fake accuracy:{:.2%},'
           ' Accuracy:{:.2%}, Average precision:{:.2%}.')
          .format(model_path, dataset_path, r_acc, f_acc, acc, ap))
    
time_end=time.time()
print('Time cost {:.2f}s.'.format(time_end-time_start))

#%%
plt.hist(np.log(y_pred[y_true==0]), label='Real', bins=100, histtype='step')
plt.hist(np.log(y_pred[y_true==1]), label='Fake', bins=100, histtype='step') 
plt.legend()
plt.xlabel('log(y_pred)')
plt.ylabel('number')
plt.show()







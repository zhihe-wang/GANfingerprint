import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchsummary import summary

#%% Model
latent_size = 32

class ConvAutoencoder(torch.nn.Module):

    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, latent_size, 3, padding=1)

        self.trans1 = torch.nn.ConvTranspose2d(latent_size, 128, 3, padding=1, stride=2, output_padding=1)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1, stride=2, output_padding=1)
        self.mp = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x) # [?, 32, 224, 224]
        x = self.mp(x) # [?, 32, 112, 112]
        x = self.conv2(x)
        x = self.relu(x) # [?, 64, 112, 112]
        x = self.mp(x) # [?, 64, 56, 56]
        x = self.conv3(x)
        x = self.relu(x) # [?, 128, 56, 56]
        x = self.mp(x) # [?, 128, 28, 28]
        x = self.conv4(x)
        x = self.relu(x) # [?, 32, 28, 28]
        x = self.mp(x) # [?, 32, 14, 14]
        return x

    def decoder(self, x):
        x = self.trans1(x)
        x = self.relu(x) # [?, 128, 28, 28]
        x = self.trans2(x)
        x = self.relu(x) # [?, 64, 56, 56]
        x = self.trans3(x)
        x = self.relu(x) # [?, 32, 112, 112]
        x = self.trans4(x)
        x = self.relu(x) # [?, 3, 224, 224]
        return x

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output
    
ae = ConvAutoencoder(latent_size).cuda()
summary(ae, input_size=(3,224,224))

#%% Train
# PARAMS
BATCH = 16
EPOCH = 1
LR = 1e-9

train_root = 'G:/Deepfake_Database/CelebA/'
transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])
train_dataset = datasets.ImageFolder(train_root, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH,
                          shuffle=True, drop_last=True)

# Loss & Optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=LR)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

# Train
log = open('./weightae/log.txt', mode='a')
ae.load_state_dict(torch.load('./weightae/step_last.pth'))
ae.train()
for epoch in range(EPOCH):
    for step, (image,label) in enumerate(train_loader):
        image = image.cuda()
        optimizer.zero_grad()

        output = ae(image)
    
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()
        if step%200 == 0 and step != 0:
           print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
           print('Step{}, loss: {:.5f}'.format(step, loss.item()))
           print('Step{}, loss: {:.5f}'.format(step, loss.item()), file=log)
           save_image(image, './imgsae/step{}_in.png'.format(step))
           save_image(output, './imgsae/step{}_out.png'.format(step))
           torch.save(ae.state_dict(),'./weightae/step_last1.pth')
           print(optimizer.param_groups[0]['lr'])
           lr_scheduler.step()
log.close()
        
#%% Transform
trans_from = '.\Dataset\Validation'
trans_to = '.\Dataset\Validation_printR'
if not os.path.exists(trans_to):
    os.mkdir(trans_to)
    for d in os.listdir(trans_from):
        os.mkdir(os.path.join(trans_to,d))

transform = transforms.ToTensor()
trans_dataset = datasets.ImageFolder(trans_from, transform)
trans_loader = DataLoader(trans_dataset, batch_size=1)

ae.load_state_dict(torch.load('./weightae/step_last.pth'))
ae.eval()
for step, (image,label) in enumerate(trans_loader):
    image = image.cuda()
    output = ae(image)
    
    old_name = trans_loader.dataset.samples[step][0]
    new_name = old_name.replace(trans_from,trans_to)

    save_image(output, new_name)








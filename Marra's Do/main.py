# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Functions
def load_data(file_path, sel_image, label=None):
    images = []; labels = []
    for file_name in os.listdir(file_path)[sel_image[0]:sel_image[1]]:
        images.append(cv2.imread(file_path + file_name))
        labels.append(label)
    return images,labels

def Marra_res(images):
    res = []
    for img1 in images:
        img2 = cv2.GaussianBlur(img1,(3,3),0.5)
        img3 = img1 - img2
        res.append(img3/np.max(img3))
    return res

def Marra_fin(images):    
    res = Marra_res(images)
    res_sum = np.zeros(np.shape(res[0]))
    for r in res:
        res_sum += r
    fin = res_sum/np.size(images,0)
    fin = fin/np.max(fin)
    return fin

def Marra_corr(X,Y):
    X -= np.mean(X, axis=(0,1))
    X /= np.std(X, axis=(0,1))
    Y -= np.mean(Y, axis=(0,1))
    Y /= np.std(Y, axis=(0,1))
    return np.sum(X*Y)/X.size

#%% Main process
if __name__ == '__main__':
    #%% Generate Marra's fingerprint
    file_path = ['../Original_c40_faces/','../Deepfake_c40_faces/',
                '../Face2Face_c40_faces/','../FaceShifter_c40_faces/',
                '../FaceSwap_c40_faces/','../NeuralTextures_c40_faces/']
    fingerprint = []
    for i in range(0,6):
        images,_ = load_data(file_path[i], (0,900))
        fingerprint.append(Marra_fin(images))
        plt.title(file_path[i])
        plt.imshow(fingerprint[i][:,:,[2,1,0]]);plt.show()
    
    #%% Test  Marra's fingerprint
    for k in range(0,len(file_path)):
        images,labels = load_data(file_path[k], (900,1000), k)
        img_res = Marra_res(images)
        predicts = np.zeros(len(labels))
        for i in range(0,len(labels)):
            prob = np.zeros(len(fingerprint))
            for j in range(0,len(fingerprint)):
                prob[j] = Marra_corr(img_res[i],fingerprint[j])
            predicts[i] = np.argmax(prob)
        accuracy = sum(predicts==labels)/len(labels)
        print('The test accuracy in {} is {:.2f}%'
              .format(file_path[k], accuracy*100))
    
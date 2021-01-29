# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, extract_face

#%% Preprocess database function
def preproc_db(video_path, num_video, num_frame):
    '''This function will create two directories: frames and faces, 
    which will contain extracted frames and faces from videos'''
    frame_path = video_path + '../frames/'
    face_path = video_path + '../faces/'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    if not os.path.exists(face_path):
        os.makedirs(face_path)
    
    video_list = os.listdir(video_path)
    mtcnn = MTCNN(select_largest=True, post_process=False)
    
    for i in range(0,num_video):
        # Extract frames from videos
        cap = cv2.VideoCapture(video_path+video_list[i])
        for j in range(0,num_frame):
            cap.set(cv2.CAP_PROP_POS_MSEC, j*1000)
            success, frame = cap.read()    
            if success:
                cv2.imwrite(frame_path+'{:0>3d}_{:0>3d}.jpg'.format(i,j), frame)
                # Extract faces from frames
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame)
                if boxes is not None:
                    extract_face(frame, boxes[0], image_size=224, save_path=\
                        face_path+'{:0>3d}_{:0>3d}.jpg'.format(i,j))
                else:
                    print('Can not find face in {:0>3d}_{:0>3d}.jpg'.format(i,j))
            else:
                print('Fail to extract frame from {} in {}s'\
                      .format(video_list[i],j))
                    
#%% Load data function
def load_data(file_dir):
    images = []; labels = []
    for file_name in os.listdir(file_dir):
        images.append(cv2.imread(file_dir + file_name))
        labels.append(0)
    return images,labels

#%% Main
if __name__ == '__main__':
    num_video = 200
    num_frame = 4

    video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
        'original_sequences/youtube/c40/videos/'
    preproc_db(video_path, num_video, num_frame)
    
    video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
        'manipulated_sequences/Deepfakes/c40/videos/'
    preproc_db(video_path, num_video, num_frame)

    video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
        'manipulated_sequences/Face2Face/c40/videos/'
    preproc_db(video_path, num_video, num_frame)

    # video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
    #     'manipulated_sequences/FaceShifter/c40/videos/'
    # preproc_db(video_path, num_video, num_frame)
    
    # video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
    #     'manipulated_sequences/FaceSwap/c40/videos/'
    # preproc_db(video_path, num_video, num_frame)

    # video_path = 'G:/Deepfake_Database/FaceForensics++_dataset/'+\
    #     'manipulated_sequences/NeuralTextures/c40/videos/'
    # preproc_db(video_path, num_video, num_frame)

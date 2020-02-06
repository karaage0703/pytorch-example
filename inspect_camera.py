#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import cv2
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import platform
pf = platform.system()
if pf == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 128) # 32 = 64(IMAGE_SIZE) / 2
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    # parse options
    parser = argparse.ArgumentParser(description='pytorch')
    parser.add_argument('-w', '--weights', default='./model/janken.pt')
    parser.add_argument('-l', '--labels', default='./model/label.txt')
    parser.add_argument('-d', '--device', default='normal_cam') # normal_cam /jetson_nano_raspi_cam

    args = parser.parse_args()

    labels = []
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    NUM_CLASSES = len(labels)
    print(labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    # print(model)

    if args.device == 'normal_cam':
        cam = cv2.VideoCapture(0)
    elif args.device == 'jetson_nano_raspi_cam':
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 \
            ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
            ! videoconvert \
            ! appsink'
        cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
    else:
        print('wrong device')
        sys.exit()

    count_max = 0
    count = 0

    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count > count_max:
            # image convert
            # reference site:
            # https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411/29
            image = capture.copy()
            image = cv2.resize(image, (64, 64))
            loader = transforms.Compose([ transforms.ToTensor()])
            image = loader(image).float()
            image = image.unsqueeze(0)
            image = image.to(device)

            start = time.time()
            output = model(image)
            preds = output.argmax(dim=1, keepdim=True)
            elapsed_time = time.time() - start

            pred_label = labels[preds]

            # Put speed
            speed_info = '%s: %f' % ('speed=', elapsed_time)
            cv2.putText(capture, speed_info , (10,50), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Put label
            cv2.putText(capture, pred_label, (10,100), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('pytorchi inspector', capture)
            count = 0

    cam.release()
    cv2.destroyAllWindows()
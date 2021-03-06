# 성능평가 코드
import numpy as np
import json
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn import metrics

model = MyNetwork().to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

#############################################
# Testing code for trained model.
#############################################
try:
    model.load_state_dict(torch.load('/content/drive/My Drive/2020Fall/인공개 팀플/results/efficient_b1_model_final_20ep.pt'))
except Exception as e:
    print(e)

tf=np.array([])
score=np.array([])
model.eval() 
with torch.no_grad():
    test_loss = 0.
    test_accuracy = 0.
    test_num_data = 0.
    precision = 0.
    recall = 0.
    f_score = 0
    for batch_idx, dictionary in enumerate(test_dataloader):
        x = dictionary['image']
        y = dictionary['diagnose']
        meta = dictionary['metadata']

        logit = model(x, meta)                                                  # Feed `x` into the network, get an output, and keep it in a variable called `logit`.
        
        loss = nn.CrossEntropyLoss()(logit, y)                                  # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
        y_pred = logit.argmax(dim=1)
        accuracy = (y_pred == y).float().mean()                                 # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
        f_y_pred = y_pred.cpu().numpy()
        f_y = y.cpu().numpy()
        sm=nn.Softmax(dim=1)
        batch_score=(sm(logit)[:,1]).cpu().numpy()
        batch_tf=f_y
        score=np.hstack([score,batch_score])
        tf=np.hstack([tf,batch_tf])
        precision += metrics.precision_score(f_y_pred, f_y) * x.shape[0]
        recall += metrics.recall_score(f_y_pred, f_y) * x.shape[0]
        f_score += metrics.f1_score(f_y_pred, f_y) * x.shape[0]
        test_loss += loss.item()*x.shape[0]
        test_accuracy += accuracy.item()*x.shape[0]
        test_num_data += x.shape[0]
    precision /= test_num_data                                                  # Calculate precision
    recall /= test_num_data                                                     # Calculate recall
    f_score /= test_num_data                                                    # Calculate F1 score
    test_loss /= test_num_data
    test_accuracy /= test_num_data
    print(f'test_loss : {test_loss:.3f}')
    print(f'test_accuracy : {test_accuracy:.3f}')
    print(f'precision : {precision:.3f}')
    print(f'recall : {recall:.3f}')
    print(f'f_score : {f_score:.3f}')
print('AUROC : ', roc_auc_score(tf,score))

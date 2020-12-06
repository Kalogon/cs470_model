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

#############################################
# Training code.
#############################################

learning_rate = 0.0001
learning_rate_decay = 0.95
epoch = 10
training_loss = 0
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.05)                      #Learning rate decay to find optimal better
start_time = time.time()
print("Training Start!")
for ep in range(epoch):                                                     #Training process.
    learning_rate = learning_rate * learning_rate_decay
    training_loss = 0
    training_accuracy = 0
    train_num_data = 0
    model.train()
    for batch_idx, dictionary in enumerate(train_dataloader):
        x = dictionary['image']                                             #  Send `x` and `y` to either cpu or gpu using `device` variable.
        y = dictionary['diagnose']
        meta = dictionary['metadata']

        logit = model(x, meta)                                              # Feed `x` into the network, get an output, and keep it in a variable called `logit`.

        accuracy = (logit.argmax(dim=1) == y).float().mean()                # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
        loss = nn.CrossEntropyLoss()(logit, y)                              # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
        optimizer.zero_grad()
        loss.backward()                                                     # backward the computed loss. 
        optimizer.step()                                                    # update the network weights. 

        training_loss += loss.item()*x.shape[0]
        training_accuracy += accuracy.item()*x.shape[0]
        train_num_data += x.shape[0]

    training_loss /= train_num_data
    training_accuracy /= train_num_data
    print(f'epoch: {ep} / training_loss : {training_loss:.3f}')
    print(f'epoch: {ep} / training_accuracy : {training_accuracy:.3f}')

    model.eval()                                                            #Validation process. It's very similar to the training part, but not calculating gradient.
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        test_num_data = 0.
        for batch_idx, dictionary in enumerate(test_dataloader):
            x = dictionary['image']
            y = dictionary['diagnose']
            meta = dictionary['metadata']

            logit = model(x, meta)
            loss = nn.CrossEntropyLoss()(logit, y)

            accuracy = (logit.argmax(dim=1) == y).float().mean()

            test_loss += loss.item()*x.shape[0]
            test_accuracy += accuracy.item()*x.shape[0]
            test_num_data += x.shape[0]

        test_loss /= test_num_data
        test_accuracy /= test_num_data
        print(f'epoch: {ep} / test_loss : {test_loss:.3f}')
        print(f'epoch: {ep} / test_accuracy : {test_accuracy:.3f}')

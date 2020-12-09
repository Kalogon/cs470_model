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
epoch = 20
training_loss = 0
train_loss_list=[]
validation_loss_list=[]
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.95)                          #Learning rate decay to find optimal better
print("시작")
for ep in range(epoch):                                                         #Training process.
    training_loss = 0
    training_accuracy = 0
    train_num_data = 0
    model.train()
    st=time.time()
    for batch_idx, dictionary in enumerate(train_dataloader):
        x = dictionary['image']                                                 #  Send `x` and `y` to either cpu or gpu using `device` variable.
        y = dictionary['diagnose']
        meta = dictionary['metadata']

        logit = model(x, meta)                                                  # Feed `x` into the network, get an output, and keep it in a variable called `logit`. 

        accuracy = (logit.argmax(dim=1) == y).float().mean()                    # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
        loss = nn.CrossEntropyLoss()(logit, y)                                  # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
        optimizer.zero_grad()
        loss.backward()                                                         # backward the computed loss. 
        optimizer.step()                                                        # update the network weights. 
        training_loss += loss.item()*x.shape[0]
        training_accuracy += accuracy.item()*x.shape[0]
        train_num_data += x.shape[0]

    scheduler.step()
    training_loss /= train_num_data
    training_accuracy /= train_num_data
    print(f'epoch: {ep} / training_loss : {training_loss:.3f}')
    print(f'epoch: {ep} / training_accuracy : {training_accuracy:.3f}')
    train_loss_list.append(training_loss)
    
    model.eval()                                                                #Validation process. It's very similar to the training part, but not calculating gradient.
    with torch.no_grad():
        validation_loss = 0.
        validation_accuracy = 0.
        validation_num_data = 0.
        for batch_idx, dictionary in enumerate(validation_dataloader):
            x = dictionary['image']
            y = dictionary['diagnose']
            meta = dictionary['metadata']

            logit = model(x, meta)
            loss = nn.CrossEntropyLoss()(logit, y)

            accuracy = (logit.argmax(dim=1) == y).float().mean()

            validation_loss += loss.item()*x.shape[0]
            validation_accuracy += accuracy.item()*x.shape[0]
            validation_num_data += x.shape[0]

        validation_loss /= validation_num_data
        validation_accuracy /= validation_num_data
        validation_loss_list.append(validation_loss)
        print(f'epoch: {ep} / validation_loss : {validation_loss:.3f}')
        print(f'epoch: {ep} / validation_accuracy : {validation_accuracy:.3f}')
        print(time.time()-st)

plt.plot(np.arange(epoch),np.array(train_loss_list))
plt.plot(np.arange(epoch),np.array(validation_loss_list))

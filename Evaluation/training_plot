#!/usr/bin/python3
import string
import matplotlib.pyplot as plt
import numpy as np


f = open("." + "/training log2.txt", "r")
lines = f.readlines()
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

for i, line in enumerate(lines):
    line = line.split(',')
    if len(line) > 4:  
        train_loss.append(float( line[1].split('=')[1].strip()[7:-1] ))
        dev_loss.append( float(line[3].split('=')[1].strip()[7:-1] ))
        train_acc.append( float(line[2].split('=')[1].strip()[:] ))
        dev_acc.append( float(line[4].split('=')[1].strip()[:] ))
    else:
        continue
        
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()

n = range(50)
ax1.plot(n, train_loss,'r', label = 'train_loss')
ax1.plot(n, dev_loss, 'r',label = 'val_loss', ls = '--')
ax2.plot(n, train_acc, 'g',label = 'train_acc')
ax2.plot(n, dev_acc, 'g',label = 'val_acc', ls = '--')

ax1.set_xlabel('Epoches', fontsize = 20)
ax1.set_ylabel('Loss', fontsize = 20)
ax2.set_ylabel('Accuracy', fontsize = 20)
ax1.legend(bbox_to_anchor=(0.38, 1.15),ncol=1,prop = {'size':14})
ax2.legend(bbox_to_anchor=(0.2, 1.15),ncol=1,prop = {'size':14})
plt.show()







f = open("." + "/training log3.txt", "r")
lines = f.readlines()
train_acc = []
dev_acc = []

for i, line in enumerate(lines):
    line = line.split(':')
    if line[0].strip() == 'Accuracy':
        if len(train_acc) < 50:
            train_acc.append( float(line[1].strip()) )
        else:
            dev_acc.append( float(line[1].strip()) )
    else:
        continue

f = open("." + "/training log1.txt", "r")
lines = f.readlines()
train_loss = []
dev_loss = []

for i, line in enumerate(lines):
    line = line.split(',')
    if len(line) > 1 and line[1].strip().split(' ')[0] == 'train_loss':    # lines including `train_loss`
        train_loss.append( float(line[1].strip().split(' ')[2][7:-1]) )
        dev_loss.append( float(line[3].strip().split(' ')[2][7:-1] ))
    else:
        continue
        
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()

ax1.plot(range(len(train_loss)), train_loss,'r', label = 'train_loss')
ax1.plot(range(len(dev_loss)), dev_loss, 'r',label = 'val_loss', ls = '--')
ax2.plot(range(len(train_acc)), train_acc, 'g',label = 'train_acc')
ax2.plot(range(len(dev_acc)), dev_acc, 'g',label = 'val_acc', ls = '--')

ax1.set_xlabel('Epoches', fontsize = 20)
ax1.set_ylabel('Loss', fontsize = 20)
ax2.set_ylabel('Accuracy', fontsize = 20)
ax1.set_ylim(0, 1.3)
ax2.set_ylim(0.4, 1.02)
ax1.legend(bbox_to_anchor=(0.38, 1.15),ncol=1,prop = {'size':14})
ax2.legend(bbox_to_anchor=(0.2, 1.15),ncol=1,prop = {'size':14})
plt.show()


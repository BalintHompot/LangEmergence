# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, os
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from time import gmtime, strftime

from copy import deepcopy
from random import sample, shuffle

# read the command line options
options = options.read()
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------
data = Dataloader(options)
numInst = data.getInstCount()

params = data.params
# append options from options to params
for key, value in options.items():
  params[key] = value

#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params)
team.train()

#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training

### split tasks into train, valid and test
task_list = [t for t in range(data.numPairTasks)]

shuffle(task_list)
num_train_tasks = 18
num_val_tasks = 1
num_test_tasks = 1
train_tasks = task_list[:num_train_tasks]
val_tasks = task_list[num_train_tasks:num_train_tasks+num_val_tasks]  
test_tasks = task_list[num_train_tasks+num_val_tasks:] 

count = 0
savePath = 'models/tasks_inter_%dH_%.4flr_%r_%d_%d.tar' %\
            (params['hiddenSize'], params['learningRate'], params['remember'],\
            options['aOutVocab'], options['qOutVocab'])

matches = {}
accuracy = {}
bestAccuracy = 0

for param in team.aBot.parameters():
    param.requires_grad = False
for param in team.qBot.parameters():
    param.requires_grad = False

for episode in range(params['num_episodes']):

    totalReward = 0
    sampled_tasks = sample(train_tasks, params['num_tasks_per_episode'])

    for task in sampled_tasks:
        ## create copy of team for inner update, and inner optimizers
        batch_task_list = torch.LongTensor([task for i in range(params['batchSize'])])      ### all tasks should be the same in an iteration with maml
        copied_team = deepcopy(team)
        for param in copied_team.aBot.parameters():
            param.requires_grad = True
        for param in copied_team.qBot.parameters():
            param.requires_grad = True


        optimizer_inner = optim.Adam([{'params': team.aBot.parameters(), \
                                'lr':params['learningRate_inner']},\
                        {'params': team.qBot.parameters(), \
                                'lr':params['learningRate_inner']}])

        # get double attribute tasks
        if 'train' not in matches:
            batchImg, batchTask, batchLabels \
                                = data.getBatch(params['batchSize'], tasks=batch_task_list)
        else:
            batchImg, batchTask, batchLabels \
                    = data.getBatchSpecial(params['batchSize'], matches['train'],\
                                                             params['negFraction'], tasks=batch_task_list)

        for inner_step in range(params['inner_steps'] - 1):


            # forward pass
            copied_team.forward(Variable(batchImg), Variable(batchTask))
            # backward pass
            batchReward = copied_team.backward(optimizer_inner, batchLabels, episode)

            # take a step by optimizer
            optimizer_inner.step()
            optimizer_inner.zero_grad()
            #--------------------------------------------------------------------------
            # switch to evaluate

        ## last inner step grads will be transferred to the main model update
        ## sampling query set
        if 'train' not in matches:
            batchImg, batchTask, batchLabels \
                                = data.getBatch(params['batchSize'], tasks=batch_task_list)
        else:
            batchImg, batchTask, batchLabels \
                    = data.getBatchSpecial(params['batchSize'], matches['train'],\
                                                             params['negFraction'], tasks=batch_task_list)

        # forward pass
        copied_team.forward(Variable(batchImg), Variable(batchTask))
        totalReward += copied_team.totalReward
        # backward pass
        batchReward = copied_team.backward(optimizer_inner, batchLabels, episode)

        ## get the stored gradients and update the original model
        ABotParamList = [p for p in team.aBot.parameters()]
        for paramInd, param in enumerate(copied_team.aBot.parameters()):
            ABotParamList[paramInd] -= params['learningRate'] * param.grad

        QBotParamList = [p for p in team.qBot.parameters()]
        for paramInd, param in enumerate(copied_team.qBot.parameters()):
            QBotParamList[paramInd] -= params['learningRate'] * param.grad
        


    
    team.evaluate()

    for dtype in ['train', 'test']:
        # get the entire batch
        img, task, labels = data.getCompleteData(dtype)
        # evaluate on the train dataset, using greedy policy
        guess, _, _ = team.forward(Variable(img), Variable(task))
        # compute accuracy for color, shape, and both
        firstMatch = guess[0].data == labels[:, 0].long()
        secondMatch = guess[1].data == labels[:, 1].long()
        matches[dtype] = firstMatch & secondMatch
        accuracy[dtype] = 100*torch.sum(matches[dtype])\
                                    /float(matches[dtype].size(0))
    # switch to train
    
    team.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100: break
    
    # save for every 100 episodes
    if episode >= 0 and episode % 100 == 0:
        team.saveModel(savePath, optimizer_inner, params)

    time = strftime("%a, %d %b %Y %X", gmtime())

    print('[%s][Episode: %.2f][Query set total reward: %.4f][Tr acc: %.2f Test acc: %.2f]' % \
                                (time, episode, totalReward,\
                                accuracy['train'], accuracy['test']))
#------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime())
replaceWith = 'final_%s' % timeStamp
finalSavePath = savePath.replace('inter', replaceWith)
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer_inner, params)
#------------------------------------------------------------------------

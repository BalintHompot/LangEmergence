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
from utilities import saveModel, load_best_results, store_results
from time import gmtime, strftime

from copy import deepcopy
from random import sample, shuffle

# read the command line options
options = options.read()

MODELNAME = 'maml'
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------
data = Dataloader(options)
numInst = data.getInstCount()

params = data.params
# append options from options to params
for key, value in options.items():
  params[key] = value

### checking and creating the folders and results files
#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params)
team.train()

#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training

### split tasks into train, valid and test, storing the split in data Object
task_list = [t for t in range(data.numPairTasks)]

shuffle(task_list)
num_train_tasks = 10
num_test_tasks = 2
train_tasks = task_list[:num_train_tasks]
test_tasks = task_list[num_train_tasks:] 
data.seenTaskList = torch.LongTensor(train_tasks)
data.unseenTaskList = torch.LongTensor(test_tasks)

count = 0
savePath = 'models/' + MODELNAME + '/' + "Remember:" + str(params['remember']) + "_AoutVocab=" + str(params['aOutVocab']) + "_QoutVocab="+ str(params['qOutVocab'])
best_results = load_best_results(MODELNAME, params)

matches = {}
accuracy = {}
matches_unseen = {}
accuracy_unseen = {}
bestAccuracy = 0

for param in team.aBot.parameters():
    param.requires_grad = False
for param in team.qBot.parameters():
    param.requires_grad = False


for episode in range(params['num_episodes']):

    totalReward = 0
    sampled_tasks = sample(train_tasks, params['num_tasks_per_episode'])

    stored_abot_params = []
    stored_qbot_params = []

    for task in sampled_tasks:
        ## create copy of team for inner update, and inner optimizers
        batch_task_list = torch.LongTensor([task for i in range(params['batchSize'])])      ### all tasks should be the same in an iteration with maml
        copied_team = deepcopy(team)
        for param in copied_team.aBot.parameters():
            param.requires_grad = True
        for param in copied_team.qBot.parameters():
            param.requires_grad = True


        optimizer_inner = optim.Adam([{'params': copied_team.aBot.parameters(), \
                                'lr':params['learningRate_inner']},\
                        {'params': copied_team.qBot.parameters(), \
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

        ## storing inner gradients
        stored_abot_params.append(copied_team.aBot.parameters())
        stored_qbot_params.append(copied_team.qBot.parameters())

    ## get the stored gradients and update the original model
    for stored_abot_param_list in stored_abot_params:
        ABotParamList = [p for p in team.aBot.parameters()]
        for paramInd, param in enumerate(stored_abot_param_list):
            ABotParamList[paramInd] -= params['learningRate'] * param.grad

    for stored_qbot_param_list in stored_qbot_params:
        QBotParamList = [p for p in team.qBot.parameters()]
        for paramInd, param in enumerate(stored_qbot_param_list):
            QBotParamList[paramInd] -= params['learningRate'] * param.grad

    ## reducing lr
    if episode+1%1000 == 0:
        params['learningRate'] /= 5
        params['learningRate_inner'] /= 5
        


    ### checking after certain episodes
    if episode%params['validation_frequency'] == 0:
        team.evaluate()

        for dtype in ['train','valid', 'test']:
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
        
        ### chack acc on unseen domains
        for dtype in ['train','valid', 'test']:
            # get the entire batch
            img, task, labels = data.getCompleteData(dtype, 'unseen')
            # evaluate on the train dataset, using greedy policy
            guess, _, _ = team.forward(Variable(img), Variable(task))
            # compute accuracy for color, shape, and both

            firstMatch = guess[0].data == labels[:, 0].long()
            secondMatch = guess[1].data == labels[:, 1].long()
            matches_unseen[dtype] = firstMatch & secondMatch
            accuracy_unseen[dtype] = 100*torch.sum(matches_unseen[dtype])\
                                        /float(matches_unseen[dtype].size(0))
            
        time = strftime("%a, %d %b %Y %X", gmtime())
        avg_unseen_acc = 0.5*accuracy_unseen['valid'].item() + 0.5*accuracy_unseen['test'].item()
        print('[%s][Episode: %.2f][Query set total reward: %.4f][SEEN TASK--Tr acc: %.2f Valid acc: %.2f Test acc: %.2f][UNSEEN TASK--Tr acc: %.2f V+Tst acc: %.2f]' % \
                        (time, episode, totalReward,\
                        accuracy['train'], accuracy['valid'], accuracy['test'], accuracy_unseen['train'],avg_unseen_acc))


    
        # save model and res if validation accuracy is the best
        if accuracy['valid'] >= best_results["valid_seen_domains"]:
            saveModel(savePath, team, optimizer_inner, params)
            new_best_results = {
                "train_seen_domains" : accuracy['train'].item(),
                "valid_seen_domains" : accuracy['valid'].item(),
                "test_seen_domains": accuracy['test'].item(),
                "train_unseen_domains": accuracy_unseen['train'].item(),
                "val+test_unseen_domains" : avg_unseen_acc
            }
            best_results = new_best_results
            store_results(new_best_results, MODELNAME, params)

        # break if train accuracy reaches 100%
        if accuracy['train'] == 100: break
        # switch to train
        
        team.train()



### save final model
saveModel(savePath, team, optimizer_inner, params)
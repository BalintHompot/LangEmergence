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
from utilities import saveModel, load_best_results, store_results
import os
# read the command line options

MODELNAME = 'original'

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

### checking and creating the folders and results files
#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params)
team.train()
optimizer = optim.Adam([{'params': team.aBot.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot.parameters(), \
                                'lr':params['learningRate']}])
#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']))
numIterPerEpoch = max(1, numIterPerEpoch)
count = 0

savePath = 'models/' + MODELNAME + '/' + "Remember:" + str(params['remember']) + "_AoutVocab=" + str(params['aOutVocab']) + "_QoutVocab="+ str(params['qOutVocab'])
best_results = load_best_results(MODELNAME, params)

matches = {}
accuracy = {}
bestAccuracy = 0
for iterId in range(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch

    # get double attribute tasks
    if 'train' not in matches:
        batchImg, batchTask, batchLabels \
                            = data.getBatch(params['batchSize'])
    else:
        batchImg, batchTask, batchLabels \
                = data.getBatchSpecial(params['batchSize'], matches['train'],\
                                                        params['negFraction'])

    # forward pass
    team.forward(Variable(batchImg), Variable(batchTask))
    # backward pass
    batchReward = team.backward(optimizer, batchLabels, epoch)

    # take a step by optimizer
    optimizer.step()
    #--------------------------------------------------------------------------

    ## checking model performannce after certain iters
    if iterId % params['validation_frequency'] == 0:
        # switch to evaluate
        team.evaluate()

        for dtype in ['train', 'valid', 'test']:
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

        # save model and res if validation accuracy is the best
        if accuracy['valid'] >= best_results["valid_seen_domains"]:
            saveModel(savePath, team, optimizer, params)
            new_best_results = {
                "train_seen_domains" : accuracy['train'].item(),
                "valid_seen_domains" : accuracy['valid'].item(),
                "test_seen_domains": accuracy['test'].item(),
                "test_unseen_domains" : 0
            }
            best_results = new_best_results
            store_results(new_best_results, MODELNAME, params)

        # break if train accuracy reaches 100%
        if accuracy['train'] == 100: break

        if iterId % 100 != 0: continue

        time = strftime("%a, %d %b %Y %X", gmtime())
        print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Train: %.2f Valid: %.2f Teest: %.2f]' % \
                                    (time, iterId, epoch, team.totalReward,\
                                    accuracy['train'], accuracy['valid'], accuracy['test']))


saveModel(savePath, team, optimizer, params)

# Script to read options
# author: satwik kottur
import argparse
import pdb
import os

# read command line arguments
def read():
    parser = argparse.ArgumentParser(description='RL Train toy example')

    # Model parameters
    parser.add_argument('-hiddenSize', default=50, type=int,\
                            help='Hidden Size for the language models')
    parser.add_argument('-embedSize', default=20, type=int,\
                            help='Embed size for words')
    parser.add_argument('-imgFeatSize', default=20, type=int,\
                            help='Image feature size for each attribute')
    parser.add_argument('-qOutVocab', default=3, type=int,\
                            help='Output vocabulary for questioner')
    parser.add_argument('-aOutVocab', default=4, type=int,\
                            help='Output vocabulary for answerer')

    parser.add_argument('-dataset', default='data/64_synthetic.json',\
                            type=str, help='Path to the dataset')
    parser.add_argument('-rlScale', default=100.0, type=float,\
                            help='Weight given to rl gradients')
    parser.add_argument('-numRounds', default=2, type=int,\
                            help='Number of rounds between Q and A')
    parser.add_argument('-remember', dest='remember', action='store_true', \
                            help='Turn on/off for ABot with memory')
    parser.add_argument('-negFraction', default=0.8, type=float,\
                            help='Fraction of negative examples in batch')

    # Optimization options
    parser.add_argument('-batchSize', default=1000, type=int,\
                            help='Batch size -- number of episodes')
    parser.add_argument('-numEpochs', default=1000000, type=int,\
                            help='Maximum number of epochs to run')
    parser.add_argument('-learningRate', default=1e-3, type=float,\
                            help='Initial learning rate')
    parser.add_argument('-useGPU', dest='useGPU', action='store_true')

    parser.add_argument('-validation_frequency', default=20, type=int,\
                        help='Frequency of checking full train and test performance')
    parser.add_argument('-num_runs', default=5, type=int,\
                        help='How many times we repeat the experiment')

    # MAML options
    parser.add_argument('-inner_steps', default=5, type=int,\
                        help='Number of inner loop steps - MAML')
    parser.add_argument('-num_tasks_per_episode', default=5, type=int,\
                        help='Number of tasks in one episode - MAML')
    parser.add_argument('-num_episodes', default=100000, type=int,\
                        help='Number of episodes - MAML')
    parser.add_argument('-learningRate_inner', default=1e-3, type=float,\
                            help='Inner learning rate')
    try:
      parsed = vars(parser.parse_args())
    except IOError as err:
      parser.error(str(err))

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in parsed.items():
      print(fmtString % keyPair)

    return parsed

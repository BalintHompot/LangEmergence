DATA_PATH='data/toy420_split_0.8.json'
VISIBLE_CUDA_DEVICES=0

# Train the agents.
python train.py -learningRate 0.01 -hiddenSize 50 -batchSize 1000 -numEpochs 100000\
                -imgFeatSize 50 -embedSize 50\
                -dataset $DATA_PATH\
                -remember\
                -aOutVocab 16 -qOutVocab 4 -useGPU

# Test the agents (from a checkpoint) and visualize the dialogs.
CHECKPOINT="models/restricted_original_tasks_inter_100H_0.0100lr_False_4_3.tar"
# python test.py $CHECKPOINT

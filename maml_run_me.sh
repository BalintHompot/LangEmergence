DATA_PATH='data/toy520_split_0.8.json'
VISIBLE_CUDA_DEVICES=0

# Train the agents.
python maml_train.py -learningRate 0.001 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -dataset $DATA_PATH\
                -aOutVocab 12 -qOutVocab 3 -useGPU


# Test the agents (from a checkpoint) and visualize the dialogs.
CHECKPOINT="models/maml_tasks_inter_100H_0.0100lr_False_4_3.tar"
# python test.py $CHECKPOINT

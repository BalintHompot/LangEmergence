DATA_PATH='data/toy420_split_0.8.json'
VISIBLE_CUDA_DEVICES=0

# Train the agents.
python maml_train.py -learningRate 0.001 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 50 -embedSize 50\
                -num_episodes 15000\
                -dataset $DATA_PATH\
                -inner_steps 0\
                -aOutVocab 4 -qOutVocab 4 -useGPU\
                -learningRate_inner 0.001\


# Test the agents (from a checkpoint) and visualize the dialogs.
CHECKPOINT="models/maml_tasks_inter_100H_0.0100lr_False_4_3.tar"
# python test.py $CHECKPOINT

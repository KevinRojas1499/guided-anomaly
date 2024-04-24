TRAIN_FLAGS="--iterations 10000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 5000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
# export NCCL_P2P_DISABLE=1
mpiexec -n 1 python scripts/classifier_train_wandb.py --fine_tuning False --data_dir ./preprocessed_new/train $TRAIN_FLAGS $CLASSIFIER_FLAGS \
    --log_dir ./checkpoints_classifier --resume_checkpoint ./checkpoints_classifier/128x128_classifier_openai.pt

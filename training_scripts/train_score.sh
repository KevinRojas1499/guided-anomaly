MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr_anneal_steps 300000 --lr 1e-4 --batch_size 32"
mpiexec -n 1 python scripts/image_train.py --data_dir ./Preprocessed/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS \
    --log_dir ./checkpoints_score --resume_checkpoint ./checkpoints_score/model040000.pt
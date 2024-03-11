MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
mpiexec -n 2 python scripts/image_train.py --data_dir OCT_DATA/processed/test/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
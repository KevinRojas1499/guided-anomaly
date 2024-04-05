MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n 1 python scripts/resample_healthy.py \
    --model_path checkpoints/score.pt \
    --classifier_path checkpoints/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
    --log_dir images/ --healthy_images_path images/samples_5x128x128x3.npz
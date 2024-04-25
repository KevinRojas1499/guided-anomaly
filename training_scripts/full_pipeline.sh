NUM_SAMPLES=10
BATCH_SIZE=10
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $NUM_SAMPLES --timestep_respacing ddim25 --use_ddim True"
SCORE_CKPT="checkpoints/score.pt"
CLASSIFIER_CKPT="checkpoints/classifier.pt"
SAVE_PATH="images/samples_${NUM_SAMPLES}x128x128x3.npz"
mpiexec -n 1 python scripts/classifier_sample.py \
    --model_path $SCORE_CKPT \
    --classifier_path $CLASSIFIER_CKPT \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
    --log_dir images/ 
python3 to_image.py --image_path $SAVE_PATH --save_path images/original_images
# mpiexec -n 1 python scripts/resample_healthy.py \
#     --model_path $SCORE_CKPT \
#     --classifier_path $CLASSIFIER_CKPT \
#     $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
#     --log_dir images/ --diseased_images_path $SAVE_PATH \
#     --healthy_images_file_name samples_healthy
# python3 to_image.py --image_path images/samples_healthy.npz --save_path images/healthy_images

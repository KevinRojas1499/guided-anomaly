MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 2 --num_samples 2 --timestep_respacing ddim25 --use_ddim True"
mpiexec -n 1 python scripts/classifier_sample.py \
    --model_path checkpoints_score/model030000.pt \
    --classifier_path checkpoints/model010000.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
    --log_dir images/ #--class_cond True
NUM_SAMPLES=50
BATCH_SIZE=12
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples $NUM_SAMPLES --timestep_respacing ddim25 --use_ddim True --classifier_scale 20.0"
SCORE_CKPT="checkpoints/score_best.pt"
CLASSIFIER_CKPT="checkpoints_classifier/classifier_best.pt"
SAVE_PATH="images/samples_${NUM_SAMPLES}x128x128x3.npz"
# MODES -> 'resample','classifier-sample','classifer-train','image-train', 'comparison' 'eval-class'
mpiexec -n 1 python run_lib.py --mode comparison \
    --model_path $SCORE_CKPT \
    --classifier_path $CLASSIFIER_CKPT \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
    --log_dir images/comparison --diseased_images_path 'OCT2017 /test_folder' \
    --num_images $NUM_SAMPLES --npz_file_name new_samples #--class_cond True
    # --healthy_images_file_name samples_healthy
    # --image_path
    # --outdir


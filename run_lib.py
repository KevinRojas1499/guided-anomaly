import argparse

import scripts.classifier_sample
import scripts.classifier_train
import scripts.resample_healthy
import scripts.classifier_train_wandb
import scripts.image_train
import full_comparison
import eval_guided_classification

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str, choices=['resample','classifier-sample','classifer-train','image-train','comparison','eval-class'])
    parser.add_argument('--log_dir',type=str)
    parser.add_argument('--diseased_images_path',type=str)
    parser.add_argument('--healthy_images_file_name',type=str)
    parser.add_argument('--num_images',type=int)
    parser.add_argument('--npz_file_name',type=str)
    parser.add_argument('--unconditional_model_path',type=str)
    
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    
    if args.mode == 'resample':
        scripts.resample_healthy.main(args)
    elif args.mode == 'classifier-sample':
        scripts.classifier_sample.main(args)
    elif args.mode == 'classifer-train':
        scripts.classifier_train.main(args)
    elif args.mode == 'image-train':
        scripts.image_train.main(args)
    elif args.mode == 'comparison':
        full_comparison.main(args)
    elif args.mode == 'eval-class':
        eval_guided_classification.main(args)

if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import numpy as np
import os
# import wandb
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from math import ceil
import scripts.resample_healthy 
import scripts.classifier_sample
from guided_diffusion import dist_util, logger
from scripts.classifier_sample import create_argparser
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

p = create_argparser()
p.add_argument('--image_path',type=str)
p.add_argument('--num_images', type=int)
p.add_argument('--outdir',type=str)


label_to_disease = {
    0: 'CNV',
    1: 'DME',
    2: 'DRUSEN',
    3: 'NORMAL'
}

def get_classifier(args):
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x):
        logits = classifier(x, torch.zeros(x.shape[0], device=x.device))
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.argmax(log_probs,dim=-1)

    return cond_fn

def to_torch_im(images_healthy):
    images_healthy_torch = torch.tensor(images_healthy,device=dist_util.dev())
    images_healthy_torch = images_healthy_torch/127.5 - 1
    images_healthy_torch = images_healthy_torch.permute((0,3,1,2))
    return images_healthy_torch


def main(args):
    print(args)
    scripts.classifier_sample.main(args)
    images_ = np.load(os.path.join(logger.get_dir(), f"{args.npz_file_name}.npz"))
    generated_images, labels = images_['arr_0'], images_['arr_1']
    
    classifier = get_classifier(args)
    images_torch = to_torch_im(generated_images)

    num_batches = ceil(generated_images.shape[0]/args.batch_size)
    batch_images = images_torch.tensor_split(num_batches)
    batch_labels = np.array_split(labels, num_batches)
    correct = 0
    for im, real_labels in zip(batch_images, batch_labels):
        predicted_labels = classifier(im).cpu().detach().numpy()
        correct += np.sum(((predicted_labels - real_labels) == 0 ))
    print(correct/len(generated_images))
    print(len(generated_images))
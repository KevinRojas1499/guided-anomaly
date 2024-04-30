import torch
import torch.nn.functional as F
import numpy as np
import os
# import wandb
from PIL import Image
import argparse
import matplotlib.pyplot as plt

import scripts.resample_healthy 
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


def from_img_to_numpy(args):
    # Function to load PNG files as NumPy arrays
    images_path = args.diseased_images_path
    png_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpeg')])
    images = []
    labels = []
    for file in png_files:
        labels.append(file.split("-")[0])
        image_path = os.path.join(images_path, file)
        image = Image.open(image_path)
        images.append(np.array(image.convert('RGB').resize((128,128))))
        if len(images) == args.num_images:
            break
    np.savez(args.log_dir, images)
    return np.array(images), labels

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
    images_diseased, real_labels = from_img_to_numpy(args)
    images_diseased_torch = to_torch_im(images_diseased)
    images_healthy, accum_grads = scripts.resample_healthy.main(args,images_diseased_torch)
    # os.system('bash training_scripts/make_healthy.sh')
    # args.image_path = 'images/samples_healthy.npz'
    # images_ = np.load(args.image_path)
    # images_healthy = images_['arr_0']
    classifier = get_classifier(args)
    accum_grads = accum_grads.permute((0,2,3,1))
    images_healthy_torch = to_torch_im(images_healthy)
    labels_healthy = classifier(images_healthy_torch).cpu().detach().numpy()
    labels_diseased = classifier(images_diseased_torch).cpu().detach().numpy()

    print(labels_healthy)
    print(labels_diseased)
    for i, (im_dis, im_he) in enumerate(zip(images_diseased, images_healthy)):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(im_dis)
        ax[0].set_title(f'{real_labels[i]} vs {label_to_disease[labels_diseased[i]]}')
        ax[1].imshow(im_he)
        ax[1].set_title(label_to_disease[labels_healthy[i]])
        ax[2].imshow(accum_grads[i].cpu().detach().numpy())
        fig.savefig(f'images/comparison/{i}.png')
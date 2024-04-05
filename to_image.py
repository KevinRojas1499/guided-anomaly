import numpy as np
import os
# import wandb
from PIL import Image
import argparse

p = argparse.ArgumentParser()
p.add_argument('--image_path',type=str)
p.add_argument('--save_path', type=str)


args = p.parse_args()

images = np.load(args.image_path)['arr_0']
# print(images['arr_0'])
print(images.shape)
# print(images.shape())
for i, imag in enumerate(images):
    im = Image.fromarray(imag)
    im.save(os.path.join(args.save_path,f'image {i}.png'))
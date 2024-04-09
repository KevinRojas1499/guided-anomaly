import numpy as np
import os
# import wandb
from PIL import Image
import argparse

p = argparse.ArgumentParser()
p.add_argument('--image_path',type=str)
p.add_argument('--save_path', type=str)

label_to_disease = {
    0: 'CNV',
    1: 'DME',
    2: 'DRUSEN',
    3: 'NORMAL'
}
args = p.parse_args()

images_ = np.load(args.image_path)
images = images_['arr_0']
labels = images_['arr_1']
print(labels)
# print(images['arr_0'])
print(images.shape)
# print(images.shape())
for i, imag in enumerate(images):
    im = Image.fromarray(imag)
    im.save(os.path.join(args.save_path,f'image_{i}_{label_to_disease[labels[i]]}.png'))
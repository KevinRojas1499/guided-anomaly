import numpy as np
# import wandb
from PIL import Image

images = np.load('images/samples_1x64x64x3.npz')
# print(images['arr_0'])

# print(images.shape())
for i, imag in enumerate(images['arr_0']):
    im = Image.fromarray(imag)
    im.save(f'image {i}.png')
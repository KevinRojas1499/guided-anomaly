import numpy as np
# import wandb
from PIL import Image

images = np.load('images/samples_12x64x64x3.npz')['arr_0']
# print(images['arr_0'])
print(images.shape)
# print(images.shape())
for i, imag in enumerate(images):
    im = Image.fromarray(imag)
    im.save(f'images/healthy_images/image {i}_healthy.png')
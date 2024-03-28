import numpy as np
# import wandb
from PIL import Image

images = np.load('samples_4_64x64x3.npz')
print(images.files)
print(images)
# print(images['arr_0'])

# print(images.shape())
for i, imag in enumerate(images['arr_0']):
    print(imag)
    im = Image.fromarray(imag)
    im.save(f'image {i}.png')
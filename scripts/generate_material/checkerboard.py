import cv2
import numpy as np
from os.path import join

from PIL import Image
#'nissl_coronal_189_199_402.png', 'ihc_coronal_189_199_402.png', 'MRI_coronal_189_199_402.png'

NSQUARES = 12
INIT_PATH = '/Users/acasamitjana/Data/tmp/allen/'
file1 = 'nissl_coronal_189_199_402.png'
file2 = 'ihc_coronal_189_199_402.png'

image1 = cv2.imread(join(INIT_PATH, file1))
image2 = cv2.imread(join(INIT_PATH, file2))

sy, sx = image1.shape[:2]
square_shape_x = sy // NSQUARES + 1
square_shape_y = sx // NSQUARES + 1
square_shape_c = min(square_shape_x, square_shape_y)

square_shape_x = sy // 2 + 1
square_shape_y = sx // 2 + 1
square_shape_r = min(square_shape_x, square_shape_y)

c = np.kron([[1, 0] * NSQUARES, [0, 1] * NSQUARES] * (NSQUARES//2), np.ones((square_shape_c, square_shape_c)))

c = c[:sy, :sx]

image = np.zeros_like(image1)
if len(image1.shape) == 3:
    for it_c in range(3):
        i1tmp = image1[..., it_c]
        i2tmp = image2[..., it_c]
        i2tmp[c==1] = i1tmp[c==1]
        image[..., it_c] = i2tmp
else:
    image2[c == 1] = image1[c == 1]
    image = image2
img = Image.fromarray(image.astype('uint8'), mode='RGB')
img.save(join(INIT_PATH, 'checkerboard_' + file1[:-4] + file2[:-4] + '.png'))




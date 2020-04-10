import numpy as np

def is_close(a, n, eps):
	return np.sum(np.abs(a - n) / np.maximum(np.ones(a.shape), np.abs(a) + np.abs(n)))

def flip_image(img):
    # Flip only horisontally, this showed more useful
    if np.random.rand() > 0.8:
        img = np.flip(img, 1)
    return img
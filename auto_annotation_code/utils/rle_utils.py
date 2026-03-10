import numpy as np

def rle_encode(mask):
    pixels = mask.flatten()
    rle = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            rle.append((int(prev), count))
            prev = pixel
            count = 1
    rle.append((int(prev), count))
    return rle

def rle_decode(rle, shape):
    flat = []
    for val, count in rle:
        flat.extend([val] * count)
    return np.array(flat, dtype=np.uint16).reshape(shape)
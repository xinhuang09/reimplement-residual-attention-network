from numpy import random

#Noise Level
def add_noise(image, noise_level=0.1):
    scale = 1 + noise_level
    noise = random.normal(loc=0.0, scale=scale)
    image += noise
    return image
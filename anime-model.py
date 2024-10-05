import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

with open('anime-model.pkl', 'rb') as file:
    model = pickle.load(file)

# Generating 36 Random Images with DCGAN

plt.figure(figsize=(10, 10))

for i in range(36):
    plt.subplot(6, 6, i + 1)
    # Generate random noise for each image
    noise = tf.random.normal([1, 300])
    mg = model.generator(noise)
    # Denormalize
    mg = (mg * 255) + 255

    mg.numpy()
    image = Image.fromarray(np.uint8(mg[0]))

    plt.imshow(image)
    plt.axis('off')

plt.show()
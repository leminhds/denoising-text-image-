import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import random
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Conv2D

NOISY_IMGS_PATH = "Noisy_Documents/noisy/"
CLEAN_IMGS_PATH = "Noisy_Documents/clean/"


X_train_noisy = []
X_train_clean = []

for file in sorted(os.listdir(NOISY_IMGS_PATH)):
    # use load_img from keras to load data
    img = load_img(NOISY_IMGS_PATH + file, color_mode='grayscale', target_size=(420, 540))
    #img_to_array to change image to numpy form
    img = img_to_array(img).astype('float32')/255
    X_train_noisy.append(img)
    
for file in sorted(os.listdir(CLEAN_IMGS_PATH)):
    img = load_img(CLEAN_IMGS_PATH + file, color_mode='grayscale', target_size=(420, 540))
    img = img_to_array(img).astype('float32')/255
    X_train_clean.append(img)
    

# convert to array
X_train_noisy = np.array(X_train_noisy)
X_train_clean = np.array(X_train_clean)


  # use the first 10 noisy images as testing images
X_test_noisy = X_train_noisy[0:10,]
X_train_noisy = X_train_noisy[10:,]

X_test_clean = X_train_clean[0:10,]
X_train_clean = X_train_clean[10:,]

# Build model
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(420,540,1), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
conv_autoencoder.fit(X_train_noisy, X_train_clean, epochs=10)

output = conv_autoencoder.predict(X_test_noisy)

 # Plot Output
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

randomly_selected_imgs = random.sample(range(X_test_noisy.shape[0]),2)

for i, ax in enumerate([ax1, ax4]):
    idx = randomly_selected_imgs[i]
    ax.imshow(X_test_noisy[idx].reshape(420,540), cmap='gray')
    if i == 0:
        ax.set_title("Noisy Images")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax2, ax5]):
    idx = randomly_selected_imgs[i]
    ax.imshow(X_test_clean[idx].reshape(420,540), cmap='gray')
    if i == 0:
        ax.set_title("Clean Images")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax3, ax6]):
    idx = randomly_selected_imgs[i]
    ax.imshow(output[idx].reshape(420,540), cmap='gray')
    if i == 0:
        ax.set_title("Output Denoised Images")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
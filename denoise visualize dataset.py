import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import random
import matplotlib.pyplot as plt

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


# look at the image
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(3, 2, figsize=(10,12))
randomly_selected_imgs = random.sample(range(X_train_noisy.shape[0]),3)
# plot noisy images on the left
for i, ax in enumerate([ax1,ax3,ax5]):    
    ax.imshow(X_train_noisy[i].reshape(420,540), cmap='gray')    
    if i == 0:        
        ax.set_title("Noisy Images", size=30)    
    ax.grid(False)    
    ax.set_xticks([])    
    ax.set_yticks([])

# plot clean images on the right
for i, ax in enumerate([ax2,ax4,ax6]):    
    ax.imshow(X_train_clean[i].reshape(420,540), cmap='gray')    
    if i == 0:        
        ax.set_title("Clean Images", size=30)    
    ax.grid(False)    
    ax.set_xticks([])    
    ax.set_yticks([])

plt.tight_layout()
plt.show()
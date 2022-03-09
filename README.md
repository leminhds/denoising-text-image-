# denoising images of text
Using Deep Convolutional Autoencoder, we will denoise images of text.

The dataset used here is an extraction of data found on https://archive.ics.uci.edu/ml/datasets/NoisyOffice

The autoencoder is built using Keras.

### Where could this model be useful?
It could be as a preprocessing step for image of text like documents, shopping receipt, etc. before feeding the image to an optical character recognition (OCR) model

In my case, this model has been used to pre-process supermarket receipts and then fed into an OCR API.

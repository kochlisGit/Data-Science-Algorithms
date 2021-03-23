from sklearn.decomposition import PCA, KernelPCA
import cv2
import numpy as np


image_filepath = 'baby_yoda_image.jpg'

# Loading image from disk.
input_image = cv2.imread(image_filepath)
print('Input image shape:', input_image.shape)


# Applies PCA transformation to image.
def image_compression(image, pca_algorithm, save_filepath=None):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    # Reshaping image to perform PCA.
    reshaped_image = np.reshape(image, (height, width*channels))
    print('Reshaped Image:', reshaped_image.shape)

    # Compressing the image.
    compressed_image = pca_algorithm.fit_transform(reshaped_image)
    print('PCA Compressed Image Shape:', compressed_image.shape)

    # Plotting images.
    approximated_image = pca_algorithm.inverse_transform(compressed_image)
    approximated_original_shape_image = cv2.convertScaleAbs(np.reshape(approximated_image, (height, width, channels)))

    cv2.imshow('Input Image', image)
    cv2.imshow('Compressed Image', approximated_original_shape_image)
    cv2.waitKey()

    # Storing the image.
    if save_filepath is not None:
        cv2.imwrite(save_filepath, approximated_original_shape_image)


# Applying PCA transformation to image. No whitening is applied to prevent further data loss.
print('\nApplying PCA')

n_components = 264
whitening = False
pca = PCA(n_components=n_components, whiten=whitening)
image_compression(input_image, pca, 'plots/pca_decompressed.jpg')

# Applying Kernel PCA transformation to image, with RBF kernel.
print('\nApplying Kernel PCA')
n_components = 1
kernel = 'rbf'
kernel_pca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True, n_jobs=-1, random_state=42)
image_compression(input_image, kernel_pca, 'plots/kernel_pca_decompressed.jpg')


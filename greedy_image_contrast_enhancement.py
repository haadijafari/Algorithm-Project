import cv2
import numpy as np


def contrast_enhancement(input_image, delta):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Calculate the size of the image
    height, width = gray_image.shape

    # Create an output image with the same dimensions as the input image
    output_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Calculate the window boundaries
            start_row = max(0, i - delta)
            end_row = min(height, i + delta)
            start_col = max(0, j - delta)
            end_col = min(width, j + delta)

            # Extract the window
            window = gray_image[start_row:end_row, start_col:end_col]

            # Calculate the mean and standard deviation of the window
            mean = np.mean(window)
            std = np.std(window)
            # print(mean)
            # print(std)

            # Calculate the enhanced pixel value using the Greedy Algorithm
            enhanced_pixel = (gray_image[i, j] - mean) * (128 / std) + 128

            # Clip the pixel value to the range [0, 255]
            enhanced_pixel = np.clip(enhanced_pixel, 0, 255)

            # Assign the enhanced pixel value to the output image
            output_image[i, j] = enhanced_pixel

    # print(window.shape)
    # print(gray_image.shape)

    return output_image


# Load the input image
image = cv2.imread('pics/blox.jpg')

# Apply local contrast enhancement with a window size of 3
enhanced_image = contrast_enhancement(image, delta=80)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.addWeighted(image, 0.75, enhanced_image, 0.25, 0)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

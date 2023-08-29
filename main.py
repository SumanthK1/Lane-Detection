from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from PIL import ImageFilter
import matplotlib.pyplot as plt

def grayscale(image):
    # Convert the image to grayscale
    return image.convert("L")

def gaussian_blur(image, kernel_size=5, sigma=1.5):
    # Create a Gaussian kernel
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2) / (2*sigma**2)), (kernel_size, kernel_size))
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Convolve the image with the kernel
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))

def sobel_filters(image):
    # Define Sobel kernels for horizontal and vertical edge detection
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convert image to numpy array
    image_array = np.array(image)

    # Convolve the image with the Sobel kernels
    gradient_x = convolve2d(image_array, kernel_x, mode='same', boundary='symm')
    gradient_y = convolve2d(image_array, kernel_y, mode='same', boundary='symm')

    return gradient_x, gradient_y

def edge_magnitude_and_direction(gradient_x, gradient_y):
    # Calculate the magnitude and direction of the edge
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)

    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    # Quantize the gradient directions into four bins: 0, 45, 90, 135 degrees
    quantized_direction = (np.round(direction * 4 / np.pi) % 4).astype(int)

    rows, cols = magnitude.shape
    suppressed_magnitude = np.zeros((rows, cols), dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if quantized_direction[i, j] == 0:
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif quantized_direction[i, j] == 1:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif quantized_direction[i, j] == 2:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            else:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

            if magnitude[i, j] >= np.max(neighbors):
                suppressed_magnitude[i, j] = magnitude[i, j]

    return suppressed_magnitude

def double_thresholding(image, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
    # Calculate the high and low thresholds
    low_threshold = np.max(image) * low_threshold_ratio
    high_threshold = np.max(image) * high_threshold_ratio

    # Initialize the edge map
    rows, cols = image.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)

    # Find strong and weak edges
    strong_edges_i, strong_edges_j = np.where(image >= high_threshold)
    weak_edges_i, weak_edges_j = np.where((image >= low_threshold) & (image < high_threshold))

    # Set strong edges in the edge map
    edge_map[strong_edges_i, strong_edges_j] = 255

    # Set weak edges in the edge map
    edge_map[weak_edges_i, weak_edges_j] = 100

    return edge_map

def edge_tracking_by_hysteresis(image, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
    # Double thresholding
    edge_map = double_thresholding(image, low_threshold_ratio, high_threshold_ratio)

    rows, cols = image.shape

    # Define 8-directional neighbors for edge tracking
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Perform edge tracking by hysteresis
    strong_edges_i, strong_edges_j = np.where(edge_map == 255)
    weak_edges_i, weak_edges_j = np.where(edge_map == 100)

    for i, j in zip(weak_edges_i, weak_edges_j):
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols and edge_map[ni, nj] == 255:
                edge_map[i, j] = 255
                break
        else:
            edge_map[i, j] = 0

    edge_map[edge_map == 100] = 0

    return edge_map
#
# if __name__ == "__main__":
#     # Load the input image
#     image_path = "highway.jpeg"
#     input_image = Image.open(image_path)
#
#     # Show grayscale image
#     plt.imshow(input_image)
#     plt.title("Input Image")
#     plt.show()
#
#     # Step 1: Convert the image to grayscale
#     gray_image = grayscale(input_image)
#
#     # Show grayscale image
#     plt.imshow(gray_image, cmap='gray')
#     plt.title("Grayscale Image")
#     plt.show()
#
#     # Step 2: Apply Gaussian blur
#     blurred_image = gaussian_blur(gray_image)
#
#     # Show blurred image
#     plt.imshow(blurred_image, cmap='gray')
#     plt.title("Blurred Image")
#     plt.show()
#
#     # Step 3: Apply Sobel filters for edge detection
#     gradient_x, gradient_y = sobel_filters(blurred_image)
#
#     # Show gradient_x and gradient_y images
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(gradient_x, cmap='gray')
#     axes[0].set_title("Gradient X")
#     axes[1].imshow(gradient_y, cmap='gray')
#     axes[1].set_title("Gradient Y")
#     plt.show()
#
#     # Step 4: Calculate edge magnitude and direction
#     magnitude, direction = edge_magnitude_and_direction(gradient_x, gradient_y)
#
#     # Show magnitude and direction images
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(magnitude, cmap='gray')
#     axes[0].set_title("Magnitude")
#     axes[1].imshow(direction, cmap='hsv')
#     axes[1].set_title("Direction")
#     plt.show()
#
#     # Step 5: Non-maximum suppression
#     suppressed_magnitude = non_maximum_suppression(magnitude, direction)
#
#     # Show suppressed magnitude image
#     plt.imshow(suppressed_magnitude, cmap='gray')
#     plt.title("Non-Maximum Suppressed Magnitude")
#     plt.show()
#
#     # Step 6: Edge tracking by hysteresis (Double thresholding + Edge tracking)
#     edge_map = edge_tracking_by_hysteresis(suppressed_magnitude)
#
#     # Show the final edge map
#     plt.imshow(edge_map, cmap='gray')
#     plt.title("Edge Tracking By Hysteresis")
#     plt.show()
#
#     # Show the final edge map
#     plt.imshow(edge_map, cmap='gray')
#     plt.title("Final Canny Edge Detected Image")
#     plt.show()

if __name__ == "__main__":
    # Load the input image
    image_path = "highway.jpeg"
    input_image = Image.open(image_path)

    # Step 1: Convert the image to grayscale
    gray_image = grayscale(input_image)

    # Step 2: Apply Gaussian blur
    blurred_image = gaussian_blur(gray_image)

    # Step 3: Apply Sobel filters for edge detection
    gradient_x, gradient_y = sobel_filters(blurred_image)

    # Step 4: Calculate edge magnitude and direction
    magnitude, direction = edge_magnitude_and_direction(gradient_x, gradient_y)

    # Step 5: Non-maximum suppression
    suppressed_magnitude = non_maximum_suppression(magnitude, direction)

    # Step 6: Edge tracking by hysteresis (Double thresholding + Edge tracking)
    edge_map = edge_tracking_by_hysteresis(suppressed_magnitude)

    # Convert numpy array back to PIL image
    output_image = Image.fromarray(edge_map)

    # Display the input and output images
    input_image.show(title="Input Image")
    output_image.show(title="Canny Edge Detected Image")

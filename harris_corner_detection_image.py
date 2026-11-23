import cv2
import numpy as np

# Load the original image
image = cv2.imread("harris_corner_detection_image.jpg")

# Check if the image loaded successfully
if image is None:
    raise FileNotFoundError("Image not found!")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to float32
gray = np.float32(gray)

# Apply Harris Corner Detection
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate the detected corners
corners = cv2.dilate(corners, None)

# Define a threshold to filter weak corners
threshold = 0.01 * corners.max()

# Create a copy of the original image
result = image.copy()

# Draw circles around detected corners
for y in range(corners.shape[0]):
    for x in range(corners.shape[1]):
        if corners[y, x] > threshold:
            cv2.circle(result, (x, y), 3, (0, 0, 255), 1)

# Save the image with detected corners
cv2.imwrite("harris_corner_detection_result.jpg", result)

# Display the original image and detected corners
cv2.imshow("Original Image", image)
cv2.imshow("Harris Corner Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
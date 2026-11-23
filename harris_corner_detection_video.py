
import cv2
import numpy as np
# Load the original video
video = cv2.VideoCapture("harris_corner_detection_image_V.mp4")

# Check if the video opened successfully
if not video.isOpened():
    raise FileNotFoundError("Video not found!")


# Read and process each frame
while True:
    ret, frame = video.read()

    # Stop when there are no more frames
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to float32
    gray = np.float32(gray)

    # Apply Harris Corner Detection
    corners = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)

    # Dilate the detected corners
    corners = cv2.dilate(corners, None)

    # Define a threshold
    threshold = 0.01 * corners.max()

    # Find strong local corner responses
    local_max = cv2.dilate(corners, np.ones((7, 7), np.uint8))

    # Keep only strong local peaks
    corner_mask = (corners > threshold) & (corners == local_max)

    # Get corner coordinates
    ys, xs = np.where(corner_mask)

    # Draw circles around detected corners
    for x, y in zip(xs, ys):
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), 1)

    # Display video with detected corners
    cv2.imshow("Harris Corner Detection - Video", frame)

    # Press q to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# Release the video and close windows
video.release()
cv2.destroyAllWindows()

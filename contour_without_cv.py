import cv2
import numpy as np
from PIL import Image, ImageOps
import time
from skimage import feature
import matplotlib.pyplot as plt

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot capture frame from camera.")
    return frame

def process_frame_with_pillow(frame):
    pil_image = Image.fromarray(frame)
    pil_image = ImageOps.grayscale(pil_image)
    return pil_image

def canny_edge_detection(image, low_threshold, high_threshold):
    image_np = np.array(image)
    edges = feature.canny(image_np, sigma=1)
    return edges

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 6))

    while True:
        start_time = time.time()

        frame = capture_frame(cap)

        pil_image = process_frame_with_pillow(frame)

        edges = canny_edge_detection(pil_image, 100, 200)

        end_time = time.time()
        frame_time = end_time - start_time

        
        ax.clear()
        ax.imshow(edges, cmap='gray')
        ax.set_title(f'Canny Edges\nTime: {frame_time:.3f}s')

        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
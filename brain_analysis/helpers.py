import cv2
import numpy as np

def detect_circles(image_path, output_path, dp=1.0, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100):
    """
    Detects circles in an image using the Hough Circle Transform and saves the result.

    :param image_path: Path to the input image file.
    :param output_path: Path where the processed image will be saved.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles using Hough Circles transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Ensure at least some circles were found
    if circles is not None:
        print(f"Detected {len(circles[0])} circles.")
        # Convert the circle parameters a, b, and r to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw a rectangle (center) to indicate the center of the circle
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Save the processed image
    cv2.imwrite(output_path, image)

import cv2
import numpy as np



def detect_circles_and_calculate_area(image_path, output_path, dp=1.0, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100):
    """
    Detects circles in an image using the Hough Circle Transform, saves the result, and calculates the combined area.

    :param image_path: Path to the input image file.
    :param output_path: Path where the processed image will be saved.
    :return: Combined area of all detected circles.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles using Hough Circles transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Ensure at least some circles were found
    if circles is not None:
        print(f"Detected {len(circles[0])} circles.")
        # Convert the circle parameters a, b, and r to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw a rectangle (center) to indicate the center of the circle
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Save the processed image
    cv2.imwrite(output_path, image)

    # Calculate the combined area of all circles
    combined_area = 0
    if circles is not None:
        radi = []
        for (x, y, r) in circles:
            radi.append((x, y, r))
            # Area of a circle is πr²
            area = np.pi * (r ** 2)
            combined_area += area

    return combined_area / 200000, radi

import cv2
import numpy as np
from rembg import remove

def remove_background_and_calculate_area(image_path):
    """
    Reads an image, removes its background, and calculates the area of the main object.

    :param image_path: Path to the input image file.
    :return: Area of the main object in the image.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert to grayscale
    # Remove background using rembg
    image_nobg = remove(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_nobg, cv2.COLOR_BGR2GRAY)

    # Thresholding to separate foreground from background
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imwrite('thresholded_image.png', threshold)  # Save the thresholded image for debugging

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area of the main object
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour) / 200000
    else:
        area = 0

    return area

import cv2

def draw_circles_on_image(image_path, circles, output_path):
    """
    Draws circles on an image based on a list of (x, y, r) tuples and saves the result.

    :param image_path: Path to the input image file.
    :param circles: List of tuples, where each tuple is (x, y, r) representing the center and radius of a circle.
    :param output_path: Path where the processed image will be saved.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (1440, 720))
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Draw each circle on the image
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Green color with thickness of 4

    # Save the processed image
    cv2.imwrite(output_path, image)

# Example usage
# circles = [(100, 100, 30), (200, 150, 40), (300, 200, 50)]
# draw_circles_on_image('path_to_your_image.jpg', circles, 'path_to_save_processed_image.jpg')


# Example usage
# area = remove_background_and_calculate_area('path_to_your_image.jpg')
# print(f"The area of the main object is: {area}")



# Example usage:
# detect_circles('path/to/input/image.jpg', 'path/to/output/Detected_Circles.png')

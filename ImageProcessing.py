import cv2
import numpy as np

def mask_cartridge_case(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define the colors for masking
    colors = {
        'breech_face': (0, 0, 255),   # Red
        'aperture_shear': (0, 255, 0),           # Green
        'firing_pin_impression': (255, 0, 255),  # Purple
        'firing_pin_drag': (255, 255, 0),        # Light Blue
        'firing_pin_drag_direction': (255, 0, 0) # Blue Arrow
    }

    # Experiment with intensity ranges for masks
    masks = {
        'breech_face': cv2.inRange(img, 180, 255),
        'aperture_shear': cv2.inRange(img, 50, 100),
        'firing_pin_impression': cv2.inRange(img, 0, 50),
        'firing_pin_drag': cv2.inRange(img, 200, 255),
        'firing_pin_drag_direction': cv2.inRange(img, 200, 255)
    }

    # Create a blank color image with 3 channels
    colored_img = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), dtype=np.uint8)

    # Apply colors to masks
    for key, color in colors.items():
        colored_img[masks[key] != 0] = color

    # Find contours in the firing pin drag mask
    contours, _ = cv2.findContours(masks['firing_pin_drag'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw a circle around the firing pin drag (largest contour)
    if contours:
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(colored_img, center, radius, colors['firing_pin_drag'], 2)

        # Draw an arrow pointing towards the tip of the bullet
        arrow_length = 30
        tip = (center[0] + arrow_length, center[1])
        cv2.arrowedLine(colored_img, center, tip, colors['firing_pin_drag_direction'], 2, tipLength=0.4)

    # Display the result
    cv2.imshow('Original Image', img)
    cv2.imshow('Masked Image', colored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/Users/heberantony/Downloads/Picture1.png'
mask_cartridge_case(image_path)

import cv2
import numpy as np
import os

def car_detection(folder_path):
    """
    Processes images in a specified folder by identifying and altering specific colored regions.

    This function iterates over PNG image files in the given folder. For each image, it identifies contours of 
    a specific blue color (representing a car), dilates these contours, and checks for the prevalence of black 
    (background) and lilac (road) colors within the dilated area. If more black than lilac pixels are found, 
    it changes the color of the original blue region to yellow. The modified image is then saved in the same folder 
    with a modified filename.

    Parameters:
    folder_path (str): The path to the folder containing the images to process.

    Returns:
    None
    """

    # Define BGR color values for different objects
    blue_bgr = np.array([142, 0, 0])  # Car
    black_bgr = np.array([0, 0, 0])   # Background
    lilac_bgr = np.array([128, 64, 128])  # Road

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Create a mask for the blue pixels
            original_blue_mask = np.all(image == blue_bgr, axis=-1)

            # Find the contours of the blue pixels
            contours, _ = cv2.findContours(original_blue_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

                kernel = np.ones((15, 15), np.uint8)  # Create a 15x15 kernel
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # Dilate the mask

                # Count the number of black and lilac pixels in the dilated mask
                black_count = np.sum(np.all(image[dilated_mask == 255] == black_bgr, axis=-1))
                lilac_count = np.sum(np.all(image[dilated_mask == 255] == lilac_bgr, axis=-1))

                # Change color of pixels if there are more black pixels than lilac pixels
                if black_count > lilac_count:
                    change_mask = np.logical_and(original_blue_mask, mask)
                    image[change_mask] = [0, 255, 255]

            # Save the modified image
            result_filename = f"{os.path.splitext(filename)[0]}_aparcado.png"
            result_path = os.path.join(folder_path, result_filename)
            cv2.imwrite(result_path, image)

if __name__ == "__main__":
    folder_path = '/content/drive/My Drive/FotosGranada/predicted_images'
    car_detection(folder_path)

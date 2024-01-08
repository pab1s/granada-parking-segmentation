from PIL import Image, ImageFile
import os
import math

Image.MAX_IMAGE_PIXELS = None  # Removes the limit on image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle potential truncation issues

def split_image(file_path, output_folder, chunk_width=256, chunk_height=256):
    """
    Split an image into smaller chunks.

    This function divides a larger image into smaller, non-overlapping chunks of specified width and height,
    and saves them to a specified output folder. If the output folder doesn't exist, it is created.

    Parameters:
    - file_path (str): Path to the source image file to be split.
    - output_folder (str): Path to the folder where the image chunks will be saved.
    - chunk_width (int, optional): Width of each chunk. Default is 256.
    - chunk_height (int, optional): Height of each chunk. Default is 256.

    Each chunk is saved in the output folder with a filename indicating its order in the splitting process.
    The chunk dimensions will be exactly as specified, except possibly for the last row or column of chunks,
    which might be smaller if the original image's dimensions are not exact multiples of the chunk dimensions.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the image
    with Image.open(file_path) as img:
        img_width, img_height = img.size

        # Calculate the number of chunks in each dimension
        x_chunks = math.ceil(img_width / chunk_width)
        y_chunks = math.ceil(img_height / chunk_height)

        # Initialize chunk number
        chunk_number = 1

        for y in range(0, y_chunks):
            for x in range(0, x_chunks):
                # Calculate the dimensions of the chunk
                left = x * chunk_width
                upper = y * chunk_height
                right = min(left + chunk_width, img_width)
                lower = min(upper + chunk_height, img_height)

                # Create the chunk
                chunk = img.crop((left, upper, right, lower))

                # Save the chunk
                chunk.save(os.path.join(output_folder, f"chunk_{chunk_number:04}.png"))
                chunk_number += 1

if __name__ == "__main__":
    split_image("D:/Documentos/DGIIM5/h50_1009_fot_042-1066_cog.tif", "granada256")

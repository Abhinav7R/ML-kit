from PIL import Image, ImageDraw, ImageFont
import os

# import cv2
# import numpy as np

# Initialize some settings
list_of_gifs = [1, 3, 5, 13, 15]
for gif in list_of_gifs:

    image_folder = f"{gif}"
    output_gif_path = f"gifs/{gif}.gif"
    
    # image_folder = "1"
    # output_gif_path = "gifs/1.gif"

    # Collect all image paths
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

    # print(image_paths)
    image_paths.sort()  # Sort the images to maintain sequence; adjust as needed

    # Initialize an empty list to store the images
    frames = []

    # Loop through each image file to add text and append to frames
    for image_path in image_paths:
        img = Image.open(image_path)

        img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

        draw = ImageDraw.Draw(img)

        bottom_right_text = "Performance Metrics are plotted against no. of epochs"

        # Calculate x, y position of the bottom-right text
        x = img.width  - 400  # 10 pixels from the right edge
        y = img.height  - 25  # 10 pixels from the bottom edge

        # Draw bottom-right text
        draw.text((x, y), bottom_right_text, fill=(0, 0, 0))

        frames.append(img)

    # Save frames as an animated GIF
    frames[0].save(output_gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
                optimize=True)

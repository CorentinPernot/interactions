import glob
import os
from datetime import datetime

# import imageio
from PIL import Image


def get_time_string():
    now = datetime.now()
    return now.strftime("%d_%H_%M_%S")


def create_gif_from_images(image_folder, output_path, duration=100):
    images = sorted(
        glob.glob(os.path.join(image_folder, "*.png")),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )

    if not images:
        raise ValueError("No images in the folder")

    # Charger les images
    frames = [Image.open(img) for img in images]

    # Sauvegarder en GIF
    frames[0].save(
        output_path + ".gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Gif saved at : {output_path}" + ".gif")

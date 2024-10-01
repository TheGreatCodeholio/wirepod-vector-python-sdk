#!/usr/bin/env python3

import os
import sys
import time
from io import BytesIO
import struct

from PIL import Image, ImageSequence

import anki_vector
from anki_vector.util import degrees

VECTOR_SCREEN_WIDTH = 1280
VECTOR_SCREEN_HEIGHT = 720
background_color = (0, 0, 0)  # Black background

def convert_pixels_to_raw_bitmap(image, opacity_percentage):
    img_width, img_height = image.size
    bitmap = []

    for y in range(img_height):
        for x in range(img_width):
            r, g, b, a = image.getpixel((x, y))

            if opacity_percentage != 100:
                r = r * opacity_percentage // 100
                g = g * opacity_percentage // 100
                b = b * opacity_percentage // 100

            # Convert to 16-bit RGB 565
            Rr = (r & 0xF8) >> 3
            Gr = (g & 0xFC) >> 2
            Br = (b & 0xF8) >> 3

            pixel_value = (Rr << 11) | (Gr << 5) | Br
            bitmap.append(pixel_value)

    return bitmap


def display_animated_gif(image_file, speed, loops, repaint_background_at_every_frame):
    try:
        image_gif = Image.open(image_file)

        bg_image = Image.new('RGBA', (VECTOR_SCREEN_WIDTH, VECTOR_SCREEN_HEIGHT), background_color)
        img_width, img_height = bg_image.size

        for loop in range(loops):
            for frame in ImageSequence.Iterator(image_gif):
                if repaint_background_at_every_frame:
                    frame_image = bg_image.copy()
                else:
                    frame_image = frame.copy()

                # Resize the image to fit Vector's screen
                frame_resized = frame_image.resize((img_width, img_height), Image.BILINEAR)

                # Convert the image to raw bitmap data
                bitmap = convert_pixels_to_raw_bitmap(frame_resized.convert('RGBA'), 100)

                # Convert the bitmap to bytes for sending to Vector's screen
                buf = BytesIO()
                for pixel in bitmap:
                    buf.write(struct.pack('<H', pixel))  # Little-endian 16-bit format

                # Display the image on Vector's screen
                duration = image_gif.info.get('duration', 100) / 1000.0 * speed
                display_face_image(buf.getvalue(), duration)
    except Exception as e:
        print(f"Error displaying GIF: {e}")


def display_face_image(screen_data, duration):
    args = anki_vector.util.parse_command_args()

    with anki_vector.Robot(args.serial) as robot:
        robot.behavior.set_head_angle(degrees(45.0))
        robot.behavior.set_lift_height(0.0)

        # Display the image on Vector's face
        robot.screen.set_screen_with_image_data(screen_data, duration)
        time.sleep(duration)


def main():
    # Load and display an animated GIF on Vector's face
    current_directory = os.path.dirname(os.path.realpath(__file__))
    gif_path = os.path.join(current_directory, "..", "face_images", "vector.gif")

    display_animated_gif(gif_path, speed=1.0, loops=3, repaint_background_at_every_frame=True)


if __name__ == "__main__":
    main()

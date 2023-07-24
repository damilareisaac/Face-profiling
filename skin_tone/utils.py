import argparse
import functools
import glob
import os
import re
import numpy as np
import string
import sys
from pathlib import Path
import cv2
from .image import process, is_black_white


def process_image(
    filename,
    image_type_setting,
    specified_palette,
    default_palette,
    specified_tone_labels,
    default_tone_labels,
    to_bw,
    new_width,
    n_dominant_colors,
    scale,
    min_nbrs,
    min_size,
    verbose,
):
    basename, extension = filename.stem, filename.suffix

    image: np.ndarray = cv2.imread(str(filename.resolve()), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"{basename}.{extension} is not found or is not a valid image."
        print(msg, file=sys.stderr)
        return {
            "basename": basename,
            "extension": extension,
            "message": msg,
        }
    is_bw = is_black_white(image)
    image_type = image_type_setting
    if image_type == "auto":
        image_type = "bw" if is_bw else "color"
    if len(specified_palette) == 0:
        skin_tone_palette = default_palette["bw" if to_bw or is_bw else "color"]
    else:
        skin_tone_palette = specified_palette

    tone_labels = (
        specified_tone_labels
        or default_tone_labels["bw" if to_bw or is_bw else "color"]
    )
    if len(skin_tone_palette) != len(tone_labels):
        raise ValueError(
            "argument -p/--palette and -l/--labels must have the same length."
        )

    try:
        records, report_images = process(
            image,
            is_bw,
            to_bw,
            skin_tone_palette,
            tone_labels,
            new_width=new_width,
            n_dominant_colors=n_dominant_colors,
            scaleFactor=scale,
            minNeighbors=min_nbrs,
            minSize=min_size,
            verbose=verbose,
        )
        return {
            "basename": basename,
            "extension": extension,
            "image_type": image_type,
            "records": records,
            "report_images": report_images,
        }
    except Exception as e:
        msg = f"Error processing image {basename}: {str(e)}"
        print(msg, file=sys.stderr)
        return {
            "basename": basename,
            "extension": extension,
            "message": msg,
        }


# @functools.cache  # Python 3.9+
@functools.lru_cache(maxsize=128)  # Python 3.2+
def alphabet_id(n):
    letters = string.ascii_uppercase
    n_letters = len(letters)
    if n < n_letters:
        return letters[n]
    _id = ""

    while n > 0:
        remainder = (n - 1) % n_letters
        _id = letters[remainder] + _id
        n = (n - 1) // n_letters

    return _id


def build_filenames(images):
    filenames = []
    valid_images = ["*.jpg", "*.gif", "*.png", "*.jpeg", "*.webp", "*.tif"]
    for name in images:
        if os.path.isdir(name):
            filenames.extend([glob.glob(os.path.join(name, i)) for i in valid_images])
        if os.path.isfile(name):
            filenames.append([name])
    filenames = [Path(f) for fs in filenames for f in fs]
    assert len(filenames) > 0, "No valid images in the specified path."
    # Sort filenames by (first) number extracted from the filename string
    filenames.sort(key=sort_file)
    return filenames


def sort_file(filename: Path):
    nums = re.findall(r"\d+", filename.stem)
    return int(nums[0]) if nums else filename


def is_windows():
    return sys.platform in ["win32", "cygwin"]

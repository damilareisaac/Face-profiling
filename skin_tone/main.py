import os
from functools import partial
from config import TONE_BLACK_WHITE, TONE_COLOURS
from utils import alphabet_id, build_filenames_from_paths
from arg_parser import build_arguments
import numpy as np
from tqdm import tqdm

from .image import process_image


def patch_asscalar(a):
    return np.asarray(a).item()


setattr(np, "asscalar", patch_asscalar)


def main():
    args = build_arguments()
    filenames = build_filenames_from_paths(args.images)
    debug: bool = args.debug
    to_bw: bool = args.black_white
    default_tone_palette = dict(color=TONE_COLOURS, bw=TONE_BLACK_WHITE)
    specified_palette: list[str] = args.palette if args.palette is not None else []
    default_tone_labels = {
        "color": [
            "C" + alphabet_id(i) for i in range(len(default_tone_palette["color"]))
        ],
        "bw": ["B" + alphabet_id(i) for i in range(len(default_tone_palette["bw"]))],
    }
    specified_tone_labels = args.labels

    for idx, ct in enumerate(specified_palette):
        if not ct.startswith("#") and len(ct.split(",")) == 3:
            r, g, b = ct.split(",")
            specified_palette[idx] = "#%02X%02X%02X" % (int(r), int(g), int(b))

    new_width = args.new_width
    n_dominant_colors = args.n_colors
    min_size = args.min_size[:2]
    scale = args.scale
    min_nbrs = args.min_nbrs
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    image_type_setting = args.image_type

    # Start
    partial_process_image = partial(
        process_image,
        image_type_setting=image_type_setting,
        specified_palette=specified_palette,
        default_palette=default_tone_palette,
        specified_tone_labels=specified_tone_labels,
        default_tone_labels=default_tone_labels,
        to_bw=to_bw,
        new_width=new_width,
        n_dominant_colors=n_dominant_colors,
        scale=scale,
        min_nbrs=min_nbrs,
        min_size=min_size,
        verbose=debug,
    )

    with tqdm(filenames, desc="Processing images", unit="images") as pbar:
        for result in map(partial_process_image, filenames):
            if "message" in result:
                pbar.update()
                continue

            basename = result["basename"]
            image_type = result["image_type"]
            records = result["records"]
            print(records)
            pbar.set_description(f"Processing {basename}")
            n_faces = len(records)
            for face_id, record in records.items():
                if face_id == "NA":
                    n_faces = 0  # Did not detect any faces
                pbar.set_postfix(
                    {
                        "Image Type": image_type,
                        "#Faces": n_faces,
                        "Face ID": face_id,
                        "Skin Tone": record[-3],
                        "Label": record[-2],
                        "Accuracy": record[-1],
                    }
                )
            pbar.update()

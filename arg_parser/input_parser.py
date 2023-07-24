import argparse


def build_arguments():
    # Setup arguments
    parser = argparse.ArgumentParser(
        description="Face Profiling",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="webcam",
        help="webcam or image",
    )

    parser.add_argument(
        "-i",
        "--images",
        nargs="+",
        default="./",
        metavar="IMAGE FILENAME",
        help="Image filename(s) to process;\n"
        'Supports multiple values separated by space, e.g., "a.jpg b.png";\n'
        'Supports directory or file name(s), e.g., "./path/to/images/ a.jpg";\n'
        "The app will search all images in current directory in default.",
    )
    parser.add_argument(
        "-p",
        "--palette",
        nargs="+",
        metavar="COLOR",
        help="Skin tone palette;\n"
        'Supports RGB hex value leading by "#" or RGB values separated by comma(,),\n'
        'E.g., "-p #373028 #422811" or "-p 255,255,255 100,100,100"',
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Skin tone labels; default values are the uppercase alphabet list.",
    )
    parser.add_argument(
        "-t",
        "--image_type",
        default="auto",
        metavar="IMAGE TYPE",
        help="Specify whether the inputs image(s) is/are colored or black/white.\n"
        'Valid choices are: "auto", "color" or "bw",\n'
        'Defaults to "auto", which will be detected automatically.',
        choices=["auto", "color", "bw"],
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Whether to output processed images, used for debugging and verification.",
    )
    parser.add_argument(
        "-bw",
        "--black_white",
        action="store_true",
        help="Whether to convert the input to black/white image(s).\n"
        "Then the app will use the black/white palette to classify the image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./",
        metavar="DIRECTORY",
        help="The path of output file, defaults to current directory.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        help="The number of workers to process the images, defaults to the number of CPUs in the system.",
        default=0,
    )

    parser.add_argument(
        "--n_colors",
        type=int,
        metavar="N",
        help="CONFIG: the number of dominant colors to be extracted, defaults to 2.",
        default=2,
    )
    parser.add_argument(
        "--new_width",
        type=int,
        metavar="WIDTH",
        help="CONFIG: resize the images with the specified width. Negative value will be ignored, defaults to 250.",
        default=250,
    )

    # For the next parameters, refer to https://stackoverflow.com/a/20805153/8860079
    parser.add_argument(
        "--scale",
        type=float,
        help="CONFIG: how much the image size is reduced at each image scale, defaults to 1.1",
        default=1.1,
    )
    parser.add_argument(
        "--min_nbrs",
        type=int,
        metavar="NEIGHBORS",
        help="CONFIG: how many neighbors each candidate rectangle should have to retain it.\n"
        "Higher value results in less detections but with higher quality, defaults to 5",
        default=5,
    )
    parser.add_argument(
        "--min_size",
        type=int,
        nargs="+",
        metavar=("WIDTH", "HEIGHT"),
        help='CONFIG: minimum possible face size. Faces smaller than that are ignored, defaults to "90 90".',
        default=(90, 90),
    )

    return parser.parse_args()

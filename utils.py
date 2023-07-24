import functools
import glob
import os
import re
import string
import sys
from pathlib import Path

from config import VALID_IMAGES


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


def build_filenames_from_paths(args_paths):
    filenames = []
    for name in args_paths:
        if os.path.isdir(name):
            filenames.extend([glob.glob(os.path.join(name, i)) for i in VALID_IMAGES])
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

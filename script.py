"""Compatibility launcher. Same as `python -m palbuddy`.

Settings that used to be edited at the top of this file (DATASET_FOLDER,
order, to_replace, max_power) now live in config.json, which is created on
first start and can be edited from the GUI's Train / Settings tabs.

    python script.py          # GUI
    python script.py --cli    # original text commands (record, train, save, infer, fastcal, ...)
"""

import sys

from palbuddy.cli import main

if __name__ == "__main__":
    sys.exit(main())

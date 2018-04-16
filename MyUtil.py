# -*- coding: utf-8 -*-
import sys
from pathlib import Path


def find_exedir():
    if getattr(sys, 'frozen', False):
         exedir = str(Path(sys.executable).parent)
    else:
         exedir = str(Path(__file__).parent)
    return exedir

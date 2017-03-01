# -*- coding: utf-8 -*-
import sys
import os
import glob


def find_exedir():
    if getattr(sys, 'frozen', False):
         exedir = os.path.dirname(sys.executable)
    else:
         exedir = os.path.dirname(__file__)
    return exedir

def find_rootdir():
    # exeの場所（管理者権限がないと書き込めずエラーになるのでHOMEに）
    if getattr(sys, 'frozen', False):
         rootdir = os.path.expanduser('~')
    else:
         rootdir = os.path.dirname(__file__)
    # ホーム
    return rootdir

def get_actual_filename(name):
    dirs = name.split('\\')
    # disk letter
    test_name = [dirs[0].upper()]
    for d in dirs[1:]:
        test_name += ["%s[%s]" % (d[:-1], d[-1])]
    res = glob.glob('\\'.join(test_name))
    if not res:
        #File not found
        return None
    return res[0].rstrip('\\')

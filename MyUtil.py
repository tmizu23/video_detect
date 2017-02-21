# -*- coding: utf-8 -*-
import sys
import os
import glob

def find_initdir():
    # exeの場所（管理者権限がないと書き込めずエラーになるのでHOMEに）
    if getattr(sys, 'frozen', False):
         #datadir = os.path.dirname(sys.executable)
         datadir = os.path.expanduser('~')
    else:
         datadir = os.path.dirname(__file__)
    # ホーム
    return datadir

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

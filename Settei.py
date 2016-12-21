# -*- coding: utf-8 -*-
import os
from os.path import join, splitext, basename
import codecs
import json
from collections import OrderedDict
import glob
from MyUtil import get_actual_filename

class Settei():
    u"""処理設定用クラス."""

    def __init__(self):
        u"""初期設定."""
        self.settings = OrderedDict([
            ("webcam", False),
            ("device", 0),
            ("playdir", get_actual_filename(os.path.dirname(__file__))),
            ("outdir", get_actual_filename(os.path.dirname(__file__))),
            ("avi", True),
            ("mov", True),
            ("mpg", True),
            ("mp4", True),
            ("wmv", True),
            ("flv", True),
            ("mts", True),
            ("m2ts", True),
            ("writevideo", True),
            ("writejpg", True),
            ("bounding", True),
            ("display", True),
            ("verbose", True),
            ("detectionTop", 0),
            ("detectionBottom", 720),
            ("detectionLeft", 0),
            ("detectionRight", 1080),
            ("speedSlider", 50)])

    def load_cui_settings(self, args):
        u"""コマンドライン引数の読み込み."""
        self.settings["writevideo"] = False
        self.settings["writejpg"] = False
        self.settings["bounding"] = False
        self.settings["avi"] = False
        self.settings["mov"] = False
        self.settings["mpg"] = False
        self.settings["mp4"] = False
        self.settings["wmv"] = False
        self.settings["flv"] = False
        self.settings["mts"] = False
        self.settings["m2ts"] = False

        if args["--inpdir"]:
            inpdir = args["--inpdir"]
            self.settings["playdir"] = inpdir.replace('/', os.sep)
        if args["--outdir"]:
            outdir = args["--outdir"]
            self.settings["outdir"] = outdir.replace('/', os.sep)
        if args["--webcam"]:
            self.settings["webcam"] = True
            self.settings["device"] = args["--webcam"]
        if args["--avi"]:
            self.settings["avi"] = args["--avi"]
        if args["--mov"]:
            self.settings["mov"] = args["--mov"]
        if args["--mpg"]:
            self.settings["mpg"] = args["--mpg"]
        if args["--mp4"]:
            self.settings["mp4"] = args["--mp4"]
        if args["--wmv"]:
            self.settings["wmv"] = args["--wmv"]
        if args["--flv"]:
            self.settings["flv"] = args["--flv"]
        if args["--mts"]:
            self.settings["mts"] = args["--mts"]
        if args["--m2ts"]:
            self.settings["m2ts"] = args["--m2ts"]
        if args["-v"]:
            self.settings["writevideo"] = True
        if args["-j"]:
            self.settings["writejpg"] = True
        if args["-b"]:
            self.settings["bounding"] = True
        if args["--area"]:
            detectarea = args["--area"].split(",")
            self.settings["detectionTOP"] = detectarea[0]
            self.settings["detectionBottom"] = detectarea[1]
            self.settings["detectionLeft"] = detectarea[2]
            self.settings["detectionRight"] = detectarea[3]
        if args["-d"]:
            self.settings["verbose"] = True

    def load_settings(self, setting_file):
        u"""設定ファイル（settings.json）の読み込み."""
        if os.path.exists(setting_file):
            f = codecs.open(setting_file, 'r', 'utf-8')  # 書き込みモードで開く
            self.settings = json.load(f)
            # outdirとplaydirの存在確認
            outdir = self.settings["outdir"]
            if not os.path.exists(outdir):
                self.settings["outdir"] = get_actual_filename(
                    os.path.dirname(__file__))
            playdir = self.settings["playdir"]
            if not os.path.exists(playdir):
                self.settings["playdir"] = get_actual_filename(
                    os.path.dirname(__file__))

    def save_settings(self, setting_file):
        u"""設定ファイルの書き出し."""
        inidir = get_actual_filename(os.path.dirname(__file__))
        f = codecs.open(join(inidir, setting_file),
                        'w', 'utf-8')  # 書き込みモードで開く
        json.dump(self.settings, f, indent=2,
                  sort_keys=False, ensure_ascii=False)
        f.close()

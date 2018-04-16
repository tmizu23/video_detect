# -*- coding: utf-8 -*-
import sys
from os.path import join
import codecs
import json
from collections import OrderedDict
from MyUtil import find_exedir
from pathlib import Path

class Settei():
    u"""処理設定用クラス."""

    def __init__(self):
        u"""初期設定."""
        self.settings = OrderedDict([
            ("webcam", False),
            ("device", 0),
            ("playdir", find_exedir()),
            ("outdir", find_exedir()),
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
            ("crop", True),
            ("display", True),
            ("verbose", True),
            ("detecttype", "detectA"),
            ("imgscale", 1.0),
            ("detectionTop", 0),
            ("detectionBottom", 720),
            ("detectionLeft", 0),
            ("detectionRight", 1080),
            ("speedSlider", 50)])

    def load_cui_settings(self, args):
        u"""コマンドライン引数の読み込み."""
        if args["--settings"]:
            if not Path(args["--settings"]).exists():
                print("Setting file is not exist.")
                sys.exit()
            self.load_settings(args["--settings"])
        if args["--inpdir"]:
            inpdir = args["--inpdir"]
            self.settings["playdir"] = str(Path(inpdir))
            if not Path(inpdir).exists():
                print("Input folder is not exist.")
                sys.exit()
        if args["--outdir"]:
            outdir = args["--outdir"]
            self.settings["outdir"] = str(Path(outdir))
            if not Path(outdir).exists():
                print("Ouput folder is not exist.")
                sys.exit()
        if args["--debug"]:
            self.settings["verbose"] = True

    def load_settings(self, setting_file):
        u"""設定ファイル（settings.json）の読み込み."""
        if Path(setting_file).exists():
            f = codecs.open(setting_file, 'r', 'utf-8')  # 書き込みモードで開く
            self.settings = json.load(f)
            # outdirとplaydirの存在確認
            outdir = self.settings["outdir"]
            if not Path(outdir).exists():
                self.settings["outdir"] = find_exedir()
            playdir = self.settings["playdir"]
            if not Path(playdir).exists():
                self.settings["playdir"] = find_exedir()

    def save_settings(self, setting_file):
        u"""設定ファイルの書き出し."""
        inidir = find_exedir()
        f = codecs.open(join(inidir, setting_file),
                        'w', 'utf-8')  # 書き込みモードで開く
        json.dump(self.settings, f, indent=2,
                  sort_keys=False, ensure_ascii=False)
        f.close()

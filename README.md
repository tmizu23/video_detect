動体検知ソフト　ver1.0beta
======================
これはビデオ映像に動体が映っているかどうかをチェックするソフトです。

インストール
------

1. WinPythonをダウンロード  
  https://sourceforge.net/projects/winpython/files/WinPython_3.6/3.6.5.0/
2. WinPythonのコマンドプロンプトを起動  
  展開してWinPython Command Prompt.exeを起動
3. OpenCVのインストール  
    コマンドプロンプトで  
    pip install opencv-contrlib-python  
    
実行
------
    python video_detect.py


出力ファイルの説明
-----
- ビデオ映像  
  - 動物を検知した部分の映像を出力します。  
  - ファイル名は、元のファイル名＋検知時トラック秒数になります。


- JPEG画像
  - 動物を検知した部分の画像を出力します。
  - ファイル名は、元のファイル名＋検知時トラック秒数になります。


開発メモ
-----
Qt Designerが動かないとき   
https://qiita.com/NoTASK/items/1644e0197c84856f607a

uiの変換
- pyuic5 video_detectUI.ui -o video_detectUI.py
- pyrcc5 video_detect.qrc -o video_detect_rc.py

exe化  

https://qiita.com/matsu_mh/items/789ae0fa7bfacbdc5829

- pip install pyinstaller
- pyinstaller video_detect.py --onefile --add-data opencv_ffmpeg340_64.dll;. 

変更履歴
------
- とりあえず公開
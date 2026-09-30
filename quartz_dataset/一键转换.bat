@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在安装依赖(仅第一次需要)...
pip install pillow -q
echo 开始转换...
python convert_via_to_yolo.py
pause

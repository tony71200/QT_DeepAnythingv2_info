@echo off
REM Tạo môi trường ảo tên .venv
python -m venv .venv

REM Kích hoạt môi trường ảo
call .venv\Scripts\activate.bat

REM Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt

REM Tạo thư mục downloads\depth_anything_v2 nếu chưa có
if not exist downloads mkdir downloads
if not exist downloads\depth_anything_v2 mkdir downloads\depth_anything_v2

REM Tải model depth_anything_v2_vitl.safetensors về đúng thư mục
set MODEL_PATH=downloads\depth_anything_v2\depth_anything_v2_vitl.safetensors
if not exist %MODEL_PATH% (
    echo Dang tai model depth_anything_v2_vitl.safetensors ...
    curl -L -o %MODEL_PATH% https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors
) else (
    echo Model da ton tai: %MODEL_PATH%
) 
@echo off
REM --- CONFIGURATION ---
REM Đường dẫn đến môi trường ảo của bạn
set VENV_PATH=D:\Project\stable-diffusion-webui\venv


echo Activating virtual environment from %VENV_PATH%...
call "%VENV_PATH%\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo Loi: Khong the kich hoat moi truong ao. Vui long kiem tra lai duong dan.
    pause
    exit /b
)

echo Starting the application...
python depth_generator_ui.py


echo Application closed.
pause 
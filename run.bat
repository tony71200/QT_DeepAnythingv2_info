@echo off
REM --- CONFIGURATION ---
REM Đường dẫn đến môi trường ảo của bạn
set VENV_PATH=D:\Project\stable-diffusion-webui\venv

REM Đường dẫn đến thư mục site-packages của Python hệ thống (nơi chứa PyQt5)
set GLOBAL_SITE_PACKAGES=C:\Users\USER\AppData\Local\Programs\Python\Python310\lib\site-packages

REM --- SCRIPT ---
set PTH_FILE=%VENV_PATH%\Lib\site-packages\global_qt5.pth

echo Creating temporary path file for PyQt5...
echo %GLOBAL_SITE_PACKAGES% > "%PTH_FILE%"
if not exist "%PTH_FILE%" (
    echo Loi: Khong the tao file .pth. Vui long kiem tra quyen truy cap.
    pause
    exit /b
)

echo Activating virtual environment from %VENV_PATH%...
call "%VENV_PATH%\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo Loi: Khong the kich hoat moi truong ao. Vui long kiem tra lai duong dan.
    del "%PTH_FILE%"
    pause
    exit /b
)

echo Starting the application...
python depth_generator_ui.py

echo Cleaning up temporary path file...
del "%PTH_FILE%"

echo Application closed.
pause 
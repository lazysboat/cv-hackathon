@echo off
SETLOCAL

echo AI vs Real Image Detector
echo ========================
echo.

REM Check if virtual environment exists
IF EXIST ai_detector_env\ (
    echo Virtual environment found.
) ELSE (
    echo Creating virtual environment...
    python setup_venv.py
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call ai_detector_env\Scripts\activate.bat

IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Virtual environment activated successfully!
echo.
echo Available commands:
echo - python ai_detector.py [image_path]
echo - python batch_detector.py [directory_path]
echo - python gui_detector.py
echo.
echo Type 'deactivate' when finished to exit the virtual environment.
echo.

REM Start a new command prompt in the virtual environment
cmd /k 
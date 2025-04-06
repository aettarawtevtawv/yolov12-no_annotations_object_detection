@echo off
:: { --> !!! Disclaimer and menu options !!! <-- }
echo [ DISCLAIMER: Ensure you have Python and pip installed before proceeding, this script has not been verified for functionallity, nor has been verified that it will work on older hardware, nor windows 7 or older, linux, (eg..) also nvidia gpus of gtx 1000 series or newer recommended, as the script will resort to ising the cpu if CUDA support is not available.
echo [ Click 1 to start the installation process.
echo [ Click 0 to close.

set /p choice=Enter your choice (1 to start, 0 to close): 

if "%choice%" == "1" (
    echo Installing Python dependencies...
    pip install torch torchvision transformers opencv-python-headless pillow ultralytics numpy
    echo Dependencies installed successfully! Feel free to ask questions in my GitHub if any errors occurred during the install.
    pause
) else if "%choice%" == "0" (
    echo Installation canceled. Exiting...
    exit
) else (
    echo Invalid choice. Please run the script again and select 1 or 0.
    pause
)

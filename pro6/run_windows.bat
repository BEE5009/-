@echo off
rem Run hand_detection.py using the project's venv (Windows cmd)
pushd "%~dp0"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    python hand_detection.py %*
) else (
    echo Virtual environment not found. Create it with:
    echo python -m venv .venv
    echo Then: .venv\Scripts\activate.bat && pip install -r requirements.txt
)
popd

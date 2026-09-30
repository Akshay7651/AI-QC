@echo off
REM Windows one-time setup: double-click or run  setup_windows.bat
REM Before this: install Python 3.11 (tick "Add to PATH") and Tesseract with Hindi (UB-Mannheim installer).
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools\fetch_models.py
python tools\selfcheck.py
echo.
echo Next time run:  .venv\Scripts\activate.bat   then   python run_qc.py --input data.xlsx --local-media D:\media --agents 12
pause

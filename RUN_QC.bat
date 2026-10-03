@echo off
rem Double-click to run the AI-QC on Input\Data.xlsx with the default agent team from config.py
rem (14 downloaders + 6 readers + 6 photo analysts). The live dashboard opens in Chrome by itself.
cd /d "%~dp0"
".venv\Scripts\python.exe" run_qc.py --input Input\Data.xlsx --output results\Result.xlsx --no-risk --discard-media %*
pause

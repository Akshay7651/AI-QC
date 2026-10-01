#!/usr/bin/env python3
"""Check that this machine can run the AI-QC: python tools/selfcheck.py"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok_all = True


def line(ok, msg, hint=""):
    global ok_all
    ok_all &= bool(ok)
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok or not hint else "   -> " + hint))


print("Python:", sys.version.split()[0])
line(sys.version_info >= (3, 10), "Python 3.10 or newer", "install Python 3.11 from python.org")
print("Packages:")
for mod, pip in [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"),
                 ("pymupdf", "pymupdf"), ("openpyxl", "openpyxl"), ("sklearn", "scikit-learn"), ("joblib", "joblib"),
                 ("pytesseract", "pytesseract"), ("httpx", "httpx"), ("tqdm", "tqdm"), ("thefuzz", "thefuzz")]:
    try:
        importlib.import_module(mod)
        line(True, mod)
    except Exception:
        line(False, mod, f"pip install {pip}   (or: pip install -r requirements.txt)")
print("Tesseract (reads the printed Hindi/English):")
exe = shutil.which("tesseract")
if not exe and os.name == "nt":
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(c).exists():
            exe = c
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = c
            except Exception:
                pass
line(exe, "tesseract program found", "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki ; Linux: sudo apt install tesseract-ocr tesseract-ocr-hin")
if exe:
    try:
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True).stdout
        line("eng" in langs, "language: English")
        line("hin" in langs, "language: Hindi", "re-run the installer and tick Hindi, or: sudo apt install tesseract-ocr-hin")
    except Exception as e:
        line(False, "tesseract runs", str(e))
print("Model files (must come with the repo):")
for f, need in [("models/cells_cnn.npz", "handwritten area/loss reader (without it handwritten values come out 'not readable')"),
                ("models/sig_clf.joblib", "signature classifier"),
                ("models/digits_cnn.npz", "handwritten digit reader (without it handwritten area/loss/dates come out 'not readable')"),
                ("photo_models/heads.pkl", "photo classifiers"), ("photo_models/form.pkl", "photo-is-form detector"),
                ("photo_models/orient.pkl", "rotation detector"), ("photo_models/crop_visible.pkl", "crop-visible detector"),
                ("photo_models/yunet.onnx", "face detector"), ("photo_models/image_classification_mobilenetv2_2022apr.onnx", "pretrained image model")]:
    line((ROOT / f).exists(), f + " - " + need, "git pull again, or run: python tools/fetch_models.py" if "onnx" in f else "git pull again")
print("CPU cores:", os.cpu_count(), "-> use --agents", min(os.cpu_count() or 4, 12))
print("\nRESULT:", "READY" if ok_all else "NOT READY - fix the FAIL lines above")
sys.exit(0 if ok_all else 1)

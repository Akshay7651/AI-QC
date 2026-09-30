"""Download the third-party ONNX models used by photo_local into photo_models/ (verifies size + sha256).
usage: python tools/fetch_models.py      (idempotent; needs outbound HTTPS to media.githubusercontent.com)
photo_local degrades gracefully (colour/texture models only, outputs marked low confidence) if the backbone is missing."""
import hashlib, os, sys, urllib.request

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "image_classification_mobilenetv2_2022apr.onnx": ("image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx", 13964571,
                                                      "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"),
    "yunet.onnx": ("face_detection_yunet/face_detection_yunet_2023mar.onnx", 232589,
                   "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(dest=None):
    dest = dest or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "photo_models")
    os.makedirs(dest, exist_ok=True)
    ok = True
    for name, (path, size, digest) in MODELS.items():
        p = os.path.join(dest, name)
        if os.path.exists(p) and os.path.getsize(p) == size and sha(p) == digest:
            print("ok     ", name); continue
        print("fetch  ", name)
        try:
            urllib.request.urlretrieve(BASE + path, p)
        except Exception as e:
            print("FAILED ", name, e); ok = False; continue
        if os.path.getsize(p) != size or sha(p) != digest:
            print("BAD CHECKSUM", name); os.remove(p); ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

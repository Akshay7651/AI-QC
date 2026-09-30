"""Extract cheap + MobileNetV2 features for all data/photos (for mining candidates and training). usage: photo_mine.py SHARD NSHARDS"""
import os, sys, glob, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
import cv2, numpy as np
cv2.setNumThreads(1)
import photo_feats as PF, photo_local as PL
MB = None

def mobilenet_logits(bgr_content):
    global MB
    if MB is None:
        MB = cv2.dnn.readNetFromONNX(os.path.join(PL._MODELS, 'image_classification_mobilenetv2_2022apr.onnx'))
    x = cv2.resize(bgr_content, (224, 224), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    MB.setInput(x.transpose(2, 0, 1)[None].copy())
    return MB.forward()[0].astype(np.float16)

def extract(path):
    img = cv2.imread(path)
    small = cv2.resize(img[: int(img.shape[0] * PF.STAMP_FRAC)], (360, 525), interpolation=cv2.INTER_AREA)
    c = cv2.resize(small, (PF.W, PF.H), interpolation=cv2.INTER_AREA)
    d = dict(file=os.path.basename(path), form_f=PF.form_features(c), field_f=PF.field_features(c), water=PF.water_features(c), logits=mobilenet_logits(small))
    d['nfaces'], d['face_conf'] = PL._faces(small)
    return d

if __name__ == '__main__':
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    fs = sorted(glob.glob('data/photos/*.jpg'))[shard::n]
    out = f'cache/photos/mine_{shard}of{n}.pkl'
    done = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    t = time.time()
    for i, f in enumerate(fs):
        b = os.path.basename(f)
        if b in done: continue
        try: done[b] = extract(f)
        except Exception as e: print('fail', b, e)
        if i % 300 == 0:
            pickle.dump(done, open(out, 'wb')); print(shard, i, len(fs), time.time() - t, flush=True)
    pickle.dump(done, open(out, 'wb')); print('done', shard, len(done))

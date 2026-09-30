"""Extract cheap + MobileNetV2 features for all data/photos (for mining candidates and training). usage: photo_mine.py SHARD NSHARDS"""
import os, sys, glob, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
import cv2, numpy as np
cv2.setNumThreads(1)
import photo_feats as PF, photo_local as PL
def extract(path):
    """Features of one photo after orientation rectification (same as inference)."""
    img = cv2.imread(path)
    small = cv2.resize(img[: int(img.shape[0] * PF.STAMP_FRAC)], (360, 525), interpolation=cv2.INTER_AREA)
    c = cv2.resize(small, (PF.W, PF.H), interpolation=cv2.INTER_AREA)
    ff = PF.form_features(c)
    d = dict(file=os.path.basename(path), form_f=ff, p_form=float(PL._form_prob([ff])[0]), rot=0)
    if d['p_form'] < 0.5:
        pr = PL._orient_probs([small])[0]
        if 1 - pr[0] > PL.ROT_TH:
            d['rot'] = [90, 180, 270][int(np.argmax(pr[1:]))]
            small = PL._rotate_back(small, d['rot'])
            c = cv2.resize(small, (PF.W, PF.H), interpolation=cv2.INTER_AREA)
        d.update(field_f=PF.field_features(c), water=PF.water_features(c), logits=PL._backbone_logits(small).astype(np.float16))
        d['nfaces'], d['face_conf'] = PL._faces(small)
    return d

if __name__ == '__main__':
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    fs = sorted(glob.glob('data/photos/*.jpg'))[shard::n]
    out = f'cache/photos/mine2_{shard}of{n}.pkl'
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

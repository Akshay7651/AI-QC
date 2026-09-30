"""Train the photo models from data/eval/photos_labels.csv. usage: python tools/photo_train.py"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, pandas as pd
import photo_feats as PF

def load_feats(cache='cache/photos/train_feats.pkl'):
    lab = pd.read_csv('data/eval/photos_labels.csv')
    if os.path.exists(cache):
        d = pickle.load(open(cache, 'rb'))
        if d['files'] == list(lab.file):
            return lab, d
    ff, fl, ori = [], [], []
    for f in lab.file:
        bgr = cv2.imread('data/photos/' + f)
        c = PF.content(bgr)
        ff.append(PF.form_features(c)); fl.append(PF.field_features(c))
        ori.append(bgr[: int(bgr.shape[0] * PF.STAMP_FRAC)])
    d = dict(files=list(lab.file), form=pd.DataFrame(ff), field=pd.DataFrame(fl), content=[cv2.resize(o, (360, 525), interpolation=cv2.INTER_AREA) for o in ori])
    pickle.dump(d, open(cache, 'wb'))
    return lab, d
if __name__ == '__main__':
    t = time.time(); lab, d = load_feats(); print('feats', time.time() - t, d['form'].shape, d['field'].shape)

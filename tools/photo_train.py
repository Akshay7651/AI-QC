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
ROTS = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
FLIPMAP = {0: 0, 90: 270, 180: 180, 270: 90}


def orient_training_set(lab, d, idx):
    X, Y, G = [], [], []
    grp = lab.file.str.split('_').str[0].values
    for i in idx:
        im = d['content'][i]
        for a, k in ((im, 0), (cv2.rotate(im, ROTS[0]), 90), (cv2.rotate(im, ROTS[1]), 180), (cv2.rotate(im, ROTS[2]), 270)):
            for aa, kk in ((a, k), (cv2.flip(a, 1), FLIPMAP[k])):
                X.append(PF.orient_features(aa)); Y.append(kk); G.append(grp[i])
    return np.array(X), np.array(Y), np.array(G)


def main():
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_predict
    lab, d = load_feats()
    grp = lab.file.str.split('_').str[0].values
    out = {'n_photos': len(lab), 'n_rows': int(len(set(grp))), 'class_counts': lab.cls.value_counts().to_dict()}
    # ---- 1. form vs field
    Xf = d['form'].values; yf = lab.is_form.values
    mk = lambda: [RandomForestClassifier(150, random_state=0, min_samples_leaf=2, n_jobs=2), ExtraTreesClassifier(150, random_state=0, min_samples_leaf=2, n_jobs=2),
                  make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))]
    P = [cross_val_predict(m, Xf, yf, groups=grp, cv=GroupKFold(5), method='predict_proba')[:, 1] for m in mk()]
    p = np.mean(P, 0); pr = p > .5; tp = int((pr & (yf == 1)).sum())
    out['form'] = dict(acc=float((pr == yf).mean()), precision=tp / max(pr.sum(), 1), recall=tp / yf.sum(), errors=np.where(pr != yf)[0].tolist(),
                       ambiguous=int(((p > .2) & (p < .8)).sum()))
    forms = mk()
    for m in forms:
        m.fit(Xf, yf)
    pickle.dump(dict(models=forms, cols=list(d['form'].columns)), open('photo_models/form.pkl', 'wb'))
    # ---- 2. orientation (synthetic rotations of the upright photos)
    rotm = lab.rotated.values == 1
    up = np.where(~rotm)[0]; rot = np.where(rotm)[0]
    X, Y, G = orient_training_set(lab, d, up)
    Xr = np.array([PF.orient_features(d['content'][i]) for i in rot]); Xu = np.array([PF.orient_features(d['content'][i]) for i in up])
    pu = np.zeros((len(up), 4)); pr_ = np.zeros((len(rot), 4)); ug = grp[up]
    for tr, te in GroupKFold(5).split(X, Y, G):
        trg = set(G[tr])
        m = ExtraTreesClassifier(150, random_state=0, min_samples_leaf=2, n_jobs=2).fit(X[tr], Y[tr])
        a = [j for j in range(len(up)) if ug[j] not in trg]
        b = [j for j in range(len(rot)) if grp[rot[j]] not in trg]
        if a and set(a) - set(np.where(pu.sum(1) > 0)[0]):
            pu[a] = m.predict_proba(Xu[a])
        if b:
            pr_[b] = m.predict_proba(Xr[b])
    isF = lab.is_form.values
    TH = 0.8
    detF = (1 - pr_[:, 0] > TH)
    out['orient'] = dict(threshold=TH, field_rotated_detected=float(detF[isF[rot] == 0].mean()), n_field_rotated=int((isF[rot] == 0).sum()),
                         upright_field_false_rot=float(((1 - pu[:, 0] > TH) & (isF[up] == 0)).sum() / (isF[up] == 0).sum()),
                         upright_form_false_rot=float(((1 - pu[:, 0] > TH) & (isF[up] == 1)).sum() / (isF[up] == 1).sum()),
                         direction_90_for_rotated=float((pr_[:, 1][isF[rot] == 0] > 0.5).mean()))
    em = ExtraTreesClassifier(150, random_state=0, min_samples_leaf=2, n_jobs=2).fit(np.vstack([X, np.repeat(Xr, 0, axis=0)]) if False else X, Y)
    pickle.dump(em, open('photo_models/orient.pkl', 'wb'))
    # form sideways rule (text lines vertical -> edge_xy high)
    fr = d['form'].edge_xy.values
    out['orient']['form_rule'] = dict(thr=0.56, upright_forms_fp=int(((fr > .56) & (isF == 1) & ~rotm).sum()), rotated_forms_tp=int(((fr > .56) & (isF == 1) & rotm).sum()), n_rot_forms=int(((isF == 1) & rotm).sum()))
    # ---- 3. crop visible vs not (non-form photos only)
    nf = np.where(isF == 0)[0]
    Xc = d['field'].values[nf]; yc = (lab.cls.values[nf] == 'S').astype(int); gc = grp[nf]
    rf = RandomForestClassifier(150, min_samples_leaf=2, random_state=0, n_jobs=2, class_weight='balanced')
    pc = cross_val_predict(rf, Xc, yc, groups=gc, cv=GroupKFold(6), method='predict_proba')[:, 1]
    prc = pc > .5
    out['crop_visible'] = dict(n=len(nf), n_crop=int(yc.sum()), acc=float((prc == yc).mean()), crop_recall=float((prc & (yc == 1)).sum() / yc.sum()),
                               noncrop_recall=float((~prc & (yc == 0)).sum() / max((yc == 0).sum(), 1)), note='only 21 non-crop field-type photos (7 scenes); weak')
    rf.fit(Xc, yc)
    pickle.dump(dict(model=rf, cols=list(d['field'].columns)), open('photo_models/crop_visible.pkl', 'wb'))
    json.dump(out, open('data/eval/photo_cv.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()

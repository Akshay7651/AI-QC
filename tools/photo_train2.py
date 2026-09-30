"""Train the field-photo heads (crop present, flooded, damage state, person-only scene, crop type) twice:
  'bb'  = logistic regression on frozen MobileNetV2 embeddings,   'cl' = random forest on classical colour/texture/water features.
Reports held-out (25% of dockets) accuracy of each alone, combined (prob. average) and how often they disagree.
usage: python tools/photo_train2.py   -> photo_models/heads.pkl, data/eval/photo_heads_report.json"""
import os, sys, glob, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import photo_local as PL

def load_mined():
    M = {}
    for f in glob.glob('cache/photos/mine2_*of3.pkl'):
        M.update(pickle.load(open(f, 'rb')))
    return {k: v for k, v in M.items() if 'logits' in v}

def xb(d): return d['logits'].astype(np.float32)
def xc(d): return np.array(list(d['field_f'].values()) + list(d['water'].values()), np.float32)
COLS_CL = None

def labels():
    app = pd.read_csv('data/eval/app_values.csv', dtype={'docket_id': str}).drop_duplicates('docket_id').set_index('docket_id')
    mined = pd.read_csv('data/eval/photos_labels_mined.csv')
    old = pd.concat([pd.read_csv('data/eval/photos_labels.csv'), pd.read_csv('data/eval/photos_labels_test.csv')])
    rows = {}
    for _, r in mined.iterrows():
        rows[r.file] = dict(src='mined', damage=r.damage, scene=r.scene, crop_present=r.crop_present, flooded=r.flooded)
    for _, r in old.iterrows():
        if r.file in rows or r.cls == 'F': continue
        rows[r.file] = dict(src='old', damage={'C': 'cut & spread'}.get(r.cls, ''), scene='person-only' if r.cls == 'H' else 'field',
                            crop_present=0 if r.cls in ('M', 'H') else 1, flooded=0)
    L = pd.DataFrame.from_dict(rows, orient='index'); L['file'] = L.index; L['dk'] = L.file.str.split('_').str[0]
    return L, app

def mk_bb(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000, class_weight='balanced'))
def mk_cl(): return ExtraTreesClassifier(80, min_samples_leaf=3, max_depth=14, random_state=0, n_jobs=2, class_weight='balanced')

def evaluate(name, X1, X2, y, g, rep, seed=0, minority=None):
    """5-fold GROUPED (by docket) out-of-fold evaluation: every photo is predicted by models that never saw its row."""
    y = np.asarray(y); classes = sorted(set(y)); ci = {k: i for i, k in enumerate(classes)}
    P = {'bb': np.zeros((len(y), len(classes))), 'cl': np.zeros((len(y), len(classes)))}
    for tr, te in GroupKFold(5).split(X1, y, g):
        for key, mdl, X in (('bb', mk_bb(), X1), ('cl', mk_cl(), X2)):
            mdl.fit(X[tr], y[tr])
            for j, k in enumerate(mdl.classes_):
                P[key][te, ci[k]] = mdl.predict_proba(X[te])[:, j]
    arr = np.array(classes); out = {}
    comb = (P['bb'] + P['cl']) / 2
    for k, p in (('bb', P['bb']), ('cl', P['cl']), ('combined', comb)):
        pred = arr[p.argmax(1)]
        out[k] = float((pred == y).mean())
        out[k + '_balanced_acc'] = float(np.mean([(pred[y == c] == c).mean() for c in classes]))
    agree = P['bb'].argmax(1) == P['cl'].argmax(1); pred = arr[comb.argmax(1)]
    out['disagree_rate'] = float(1 - agree.mean())
    out['acc_when_agree'] = float((pred == y)[agree].mean()) if agree.any() else None
    out['acc_when_disagree'] = float((pred == y)[~agree].mean()) if (~agree).any() else None
    out['majority_baseline'] = float(pd.Series(y).value_counts(normalize=True).max())
    out['n'] = int(len(y)); out['classes'] = [str(c) for c in classes]
    out['per_class_recall'] = {str(k): float((pred[y == k] == k).mean()) for k in classes}
    out['per_class_precision'] = {str(k): (float((y[pred == k] == k).mean()) if (pred == k).any() else None) for k in classes}
    out['per_class_n'] = {str(k): int((y == k).sum()) for k in classes}
    rep[name] = out
    final = dict(bb=mk_bb().fit(X1, y), cl=mk_cl().fit(X2, y), classes=classes)
    return final


def main():
    M = load_mined(); L, app = labels()
    L = L[L.file.isin(M)].copy(); print('labelled with features', len(L), L.src.value_counts().to_dict())
    X1 = np.array([xb(M[f]) for f in L.file]); X2 = np.array([xc(M[f]) for f in L.file]); g = L.dk.values
    rep = {}; heads = {}
    heads['cols_cl'] = list(M[L.file.iloc[0]]['field_f'].keys()) + list(M[L.file.iloc[0]]['water'].keys())
    heads['scene'] = evaluate('scene_person_only_vs_field', X1, X2, L.scene.values, g, rep)
    F = L[L.scene == 'field']; i = L.index.get_indexer(F.index)
    heads['crop_present'] = evaluate('crop_present', X1[i], X2[i], F.crop_present.astype(int).values, g[i], rep)
    heads['flooded'] = evaluate('flooded', X1[i], X2[i], F.flooded.astype(int).values, g[i], rep)
    D = F[((F.damage != '') & (F.src == 'mined')) | (F.damage == 'cut & spread')]
    cnt = D.damage.value_counts(); keep = cnt[cnt >= 5].index; D = D[D.damage.isin(keep)]; j = L.index.get_indexer(D.index)
    rep['damage_classes_used'] = {str(k): int(v) for k, v in cnt.items()}
    heads['damage'] = evaluate('damage_state', X1[j], X2[j], D.damage.values, g[j], rep)
    # crop type: weak labels from the declared crop of every non-form, non-person photo in the pool
    files = [f for f, d in M.items() if d['nfaces'] == 0]
    crop = [app.crop_name.get(f.split('_')[0]) for f in files]
    ok = [k for k, c in enumerate(crop) if isinstance(c, str)]
    short = {'Cotton (Kapas)': 'cotton', 'Pearl Millet (Bajra)': 'pearl millet', 'Green Gram (Moong) - IR': 'green gram', 'Paddy (Dhan)': 'paddy'}
    yc = np.array([short.get(crop[k], 'other') for k in ok]); Xb = np.array([xb(M[files[k]]) for k in ok]); Xc = np.array([xc(M[files[k]]) for k in ok]); gc = np.array([files[k].split('_')[0] for k in ok])
    rep['crop_type_label_counts'] = pd.Series(yc).value_counts().to_dict()
    heads['crop_type'] = evaluate('crop_type_weak_declared', Xb, Xc, yc, gc, rep)
    import joblib; joblib.dump(heads, 'photo_models/heads.pkl', compress=3)
    json.dump(rep, open('data/eval/photo_heads_report.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

if __name__ == '__main__':
    main()

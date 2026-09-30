"""Evaluate photo_local.analyse on a labelled photo set. usage: python tools/photo_eval.py [test|train] [--noform]"""
import os, sys, time, resource, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
import numpy as np, pandas as pd
import photo_local as PL

which = sys.argv[1] if len(sys.argv) > 1 else 'test'
use_form = '--noform' not in sys.argv
lab = pd.read_csv('data/eval/photos_labels_test.csv' if which == 'test' else 'data/eval/photos_labels.csv')
app = pd.read_csv('data/eval/app_values.csv', dtype={'docket_id': str}).drop_duplicates('docket_id').set_index('docket_id')
lab['dk'] = lab.file.str.split('_').str[0]
res = {}
cpu0 = time.process_time(); r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.time()
rows = 0
for dk, g in lab.groupby('dk', sort=False):
    files = ['data/photos/' + f for f in g.file]
    fi = f'data/forms/{dk}.jpg'
    out = PL.analyse(files, app.loc[dk].to_dict(), fi if (use_form and os.path.exists(fi)) else None)
    res[dk] = out; rows += 1
r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
cpu = time.process_time() - cpu0 + (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime)
rep = dict(set=which, rows=rows, photos=len(lab), cpu_s_per_row=cpu / rows, wall_s_per_row=(time.time() - w0) / rows, used_form_image=use_form)
pf, pp, pr_, cls_pred = [], [], [], []
stamp_ok = []; dist = []
for dk, g in lab.groupby('dk', sort=False):
    o = res[dk]
    if o.get('photo_status') != 'OK':
        continue
    for k, (_, r) in enumerate(g.iterrows()):
        pf.append(o['photo_is_form_each'][k]); pp.append(o['person_each'][k] if 'person_each' in o else None)
        stamp_ok.append(o['stamp_read'][k]); dist.append(o['stamp_dist_each_m'][k])
lab = lab[lab.dk.isin([d for d, o in res.items() if o.get('photo_status') == 'OK'])].copy()
lab['pred_form'] = pf; lab['pred_person'] = pp; lab['stamp_ok'] = stamp_ok; lab['dist'] = dist
y = lab.is_form.values == 1; p = lab.pred_form.values.astype(bool)
tp = (y & p).sum()
rep['form'] = dict(precision=tp / max(p.sum(), 1), recall=tp / y.sum(), acc=(y == p).mean(), fp=int((~y & p).sum()), fn=int((y & ~p).sum()),
                   fp_files=lab.file[~y & p].tolist(), fn_files=lab.file[y & ~p].tolist())
y = lab.person.values == 1; p = lab.pred_person.values.astype(bool)
tp = (y & p).sum()
rep['person'] = dict(precision=tp / max(p.sum(), 1), recall=tp / max(y.sum(), 1), n_pos=int(y.sum()), fp=int((~y & p).sum()), fn=int((y & ~p).sum()))
rep['stamp'] = dict(parse=float(lab.stamp_ok.mean()), within20m=float((lab.dist < 20).mean()), within200m=float((lab.dist < 200).mean()), far=int((lab.dist >= 200).sum()))
json.dump(dict(rep=rep), open(f'data/eval/photo_eval_{which}.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
pickle_out = {k: v for k, v in res.items()}
import pickle; pickle.dump(pickle_out, open(f'/tmp/photo_eval_{which}.pkl', 'wb'))

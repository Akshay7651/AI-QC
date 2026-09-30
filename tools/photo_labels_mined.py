"""Hand labels for the mined candidate photos (data/eval/mined_sel.csv: 300 photos picked by heuristics from data/photos and
labelled from contact sheets). Codes: H healthy crop, P partly damaged crop, S submerged/flooded, D dried-burnt, V harvested,
B bare soil, W weeds / uncultivated land (no insured crop), p person-only scene.  (L lodged, C cut&spread: none seen.)
Output data/eval/photos_labels_mined.csv: file, tag, damage, scene, crop_present, flooded, rotated, person."""
import csv, os
T = (['P P H S H H W P W W H W H H W H W D D H H W H H P W S W H H P D H H W D', 'P W W H P H H D W H H D P H W S H P P B H P H H P H p H H H H H H H H H', 'W W P H B D H W P W H H H D D D H H H H H P H D D P H P D S D H H D W D', 'D D D P D P D D D D D W D D D P W D P D D D D P P D W B D P H D D P D W', 'W B P H B H D S P H H B P P P H P H p B H p P B P p W p P P H H H H W p', 'W W H W P H H p W W W p p W p p H H H H D P p W W p S D D W p H W W W H', 'P B P H W V P H H P H H H H H H W W H W W W W H H W H W H D H H D H H H', 'W H W D W W W W P H H D H P H H H W S W H H H W H H S S H W H W H H W D', 'H H P W H H P H H P H H'])
T = " ".join(T).split()
assert len(T) == 300, len(T)
ROT = set([4,5,13,15,19,25,29,32,33,34,35,42,46,47,64,67,68,71,75,83,86,87,88,90,91,94,95,96,97,100,101,102,113,114,117,123,126,130,131,132,133,138,141,149,154,155,164,239,268,273,274,276,279,280,282,293,294,295])
PERS = set([12,14,20,62,162,165,169,171,179,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,202,203,204,205,206,208,209,210,43,112,118,263,269])
FLOOD = set([3,26,51,101,151,270,278,279,206])
sel = [l.strip().split(',') for l in open('data/eval/mined_sel.csv')][1:]
ND = {'H': 'healthy', 'P': 'partly damaged', 'S': 'submerged', 'D': 'dried-burnt', 'V': 'harvested', 'B': 'bare soil', 'W': 'weeds-uncultivated', 'p': ''}
with open('data/eval/photos_labels_mined.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['file', 'tag', 'damage', 'scene', 'crop_present', 'flooded', 'rotated', 'person'])
    for i, (fn, tag) in enumerate(sel):
        c = T[i]
        scene = 'person-only' if c == 'p' else 'field'
        cp = 1 if c in 'HPSD' else 0     # crop present (dried = crop present but dead)
        w.writerow([fn, tag, ND[c], scene, cp, int(i in FLOOD or c == 'S'), int(i in ROT), int(i in PERS or c == 'p')])
from collections import Counter; print(Counter(T))

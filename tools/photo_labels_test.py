"""Builds data/eval/photos_labels_test.csv: held-out labels for 110 random rows (330 photos), indices into
data/eval/photo_snapshot_test.txt, labelled by viewing contact sheets. Same classes as photo_labels_build.py plus
C = cut & spread (harvest residue)."""
import csv, os
snap=[l.strip() for l in open('data/eval/photo_snapshot_test.txt')]
def R(s):
    o=[]
    for t in s.split(','):
        a,_,b=t.partition('-'); o+=range(int(a),int(b or a)+1)
    return o
F=R("3-11,15-17,21-23,27-29,33-35,39-41,51-56,60-71,75-86,90-92,105-107,114-116,120-143,144-155,159-164,189-206,210-218,222-224,228-230,243-245,255-281,285-287,288-290,294-296,300-305,309-311,315-317,321-326")
H=R("12-14")
M=R("102-104,246")
C=R("234-236")
P=set(R("12-14,37,240-242"))
ROT=set(R("0-2,171-173,181,186-188,306"))
cls={i:'S' for i in range(len(snap))}
for i in M: cls[i]='M'
for i in C: cls[i]='C'
for i in H: cls[i]='H'
for i in F: cls[i]='F'
assert len(snap)==330
with open('data/eval/photos_labels_test.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['file','cls','is_form','person','rotated'])
    for i,p in enumerate(snap): w.writerow([os.path.basename(p),cls[i],int(cls[i]=='F'),int(i in P),int(i in ROT)])
from collections import Counter; print(Counter(cls.values()),len(P),len(ROT))

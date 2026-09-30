"""Builds data/eval/photos_labels.csv from hand labels (indices into data/eval/photo_snapshot.txt).
cls: F=photo of paper form, S=field photo (crop present), M=field/other photo without a clear crop (weeds, wall, hand),
H=person/selfie (not field). Labels made by viewing contact sheets (tools/photo_sheet.py)."""
import csv, os
snap=[l.strip() for l in open('data/eval/photo_snapshot.txt')]
def R(s):
    o=[]
    for t in s.split(','):
        a,_,b=t.partition('-'); o+=range(int(a),int(b or a)+1)
    return o
F=R("0-11,15-20,33-41,48-50,54-56,75-83,87-95,105-106,110-115,119-124,128-130,134-143,144-145,164-193,200-217,224-232,236-238,242-250,263-265,272-274,278-284,291-293,300-302,309-311,315-317,321-323,327-329,333-344,348-350,357-426")
F=[i for i in F if i not in R("397-399,409-414")]
M=R("42-47,72-74,149,151,287")
H=R("12-14,254-256,430-432")
P=set(R("12-14,85,109,132,254-256,286,430-432"))
ROT=set(R("18-20,24-32,63-65,84-86,221-223,233-235,239,275-277,294-296,351-353"))
cls={}
for i in range(len(snap)): cls[i]='S'
for i in M: cls[i]='M'
for i in H: cls[i]='H'
for i in F: cls[i]='F'
with open('data/eval/photos_labels.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['file','cls','is_form','person','rotated'])
    for i,p in enumerate(snap): w.writerow([os.path.basename(p),cls[i],int(cls[i]=='F'),int(i in P),int(i in ROT)])
from collections import Counter; print(Counter(cls.values()),len(P),len(ROT))

"""Contact sheets of data/photos for hand-labelling. usage: photo_sheet.py START N OUT [stampcrop]"""
import sys, glob, os
from PIL import Image, ImageDraw
def sheet(files, out, cols=6, tw=200, th=356, offset=0):
    rows=(len(files)+cols-1)//cols
    S=Image.new('RGB',(cols*tw,rows*(th+14)),'white'); d=ImageDraw.Draw(S)
    for i,f in enumerate(files):
        try: im=Image.open(f).convert('RGB'); im.thumbnail((tw,th))
        except Exception: continue
        x,y=(i%cols)*tw,(i//cols)*(th+14); S.paste(im,(x,y+14)); d.text((x+2,y+1),str(offset+i),fill='red')
    S.save(out,quality=80)
if __name__=='__main__':
    fs=sorted(glob.glob('data/photos/*.*')); a,n=int(sys.argv[1]),int(sys.argv[2])
    sheet(fs[a:a+n],sys.argv[3],offset=a)

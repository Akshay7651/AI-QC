import pandas as pd, json, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

xlsx = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\akshay\Downloads\Level_1_QC_Requred.xlsx'

# Above this row count we skip embedding the data straight into the HTML file
# (a 100k+ row embed makes a 100MB+ .html that is slow to open). Instead index.html
# ships empty and you load the file with the "Choose File" button — it is then cached
# in the browser's IndexedDB so a refresh doesn't need the file again.
EMBED_ROW_LIMIT = 60000

print(f"Loading: {xlsx}")
d = pd.read_excel(xlsx, header=None, dtype=str).fillna('')
n = len(d)
tpl = open('tpl.html', encoding='utf-8').read()
if n <= EMBED_ROW_LIMIT:
    s = json.dumps(d.values.tolist(), ensure_ascii=False)
    out = tpl.replace('__DATA__', s)
    msg = f"Done -> index.html  ({n} rows, embedded)"
else:
    out = tpl.replace('__DATA__', 'null')
    msg = f"Done -> index.html  ({n} rows -- NOT embedded, use Choose File to load {os.path.basename(xlsx)} in the browser)"
open('index.html', 'w', encoding='utf-8').write(out)
print(msg)

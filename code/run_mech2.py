import os, sys, traceback
os.chdir('/data/yash/tmp/claude-1000/-data-yash-comp/dcc873cb-83b0-4a0a-b1ac-c9a359584756/scratchpad')
sys.path.insert(0,'.')
import pandas as pd, numpy as np
from multiprocessing import Pool
import mech2

D = pd.read_csv('labels_fam.csv')

def work(a):
    i, cid, seq = a
    try:
        return i, mech2.analyse2(f'cif/{cid}.cif', seq)
    except Exception:
        return i, {'err': traceback.format_exc()[-160:]}

if __name__ == '__main__':
    jobs=[(i,r.id,r.seq) for i,r in D.iterrows()]
    out={}
    with Pool(52) as p:
        for n,(i,r) in enumerate(p.imap_unordered(work,jobs,chunksize=3),1):
            out[i]=r
            if n%100==0: print(f"{n}/{len(jobs)}",flush=True)
    rec=[]
    for i in range(len(D)):
        d={'idx':i}; v=out.get(i)
        if isinstance(v,dict) and 'err' not in v: d.update(v)
        rec.append(d)
    M=pd.DataFrame(rec).set_index('idx')
    R=pd.concat([D,M],axis=1)
    R.to_csv('mech2_results.csv',index=False)
    print("done rows:",len(R),"failed:",sum(1 for v in out.values() if v is None or (isinstance(v,dict) and 'err' in v)))

import pandas as pd, numpy as np, itertools, collections
D = pd.read_csv('labels.csv'); D = D[D.binding.notna()].reset_index(drop=True)
seqs = D.seq.tolist()
K = 5
prof = [set(s[i:i+K] for i in range(len(s)-K+1)) for s in seqs]
n = len(seqs)
# greedy single-linkage clustering on k-mer Jaccard (proxy for sequence identity)
THRESH = 0.30
parent = list(range(n))
def find(a):
    while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
    return a
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[rb]=ra
# invert index: kmer -> designs, to avoid full O(n^2)
inv=collections.defaultdict(list)
for i,p in enumerate(prof):
    for km in p: inv[km].append(i)
cand=collections.Counter()
for km,lst in inv.items():
    if len(lst)>200: continue
    for a,b in itertools.combinations(lst,2): cand[(a,b)]+=1
for (a,b),shared in cand.items():
    j = shared/ (len(prof[a])+len(prof[b])-shared)
    if j>=THRESH: union(a,b)
fam=[find(i) for i in range(n)]
D['family']=pd.factorize(fam)[0]
D.to_csv('labels_fam.csv',index=False)
y=D.binding.astype(str).str.lower().eq('true').astype(int)
vc=D.family.value_counts()
print(f"{n} designs -> {D.family.nunique()} families")
print(f"singletons: {(vc==1).sum()}  largest: {vc.head(8).tolist()}")
big=D[D.family.isin(vc[vc>=8].index)]
g=big.groupby('family').apply(lambda x: pd.Series({'n':len(x),'hits':y[x.index].sum(),'rate':y[x.index].mean(),'len':x.seq.str.len().median(),'authors':x.author.nunique()}))
print("\nlargest families:"); print(g.sort_values('n',ascending=False).head(12).to_string(float_format=lambda v:f"{v:.2f}"))

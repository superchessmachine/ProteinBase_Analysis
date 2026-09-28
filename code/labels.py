import pandas as pd, json
df = pd.read_csv('/data/yash/comp/proteinbase_all_data_28_01_2026.csv')
rows=[]
for r in df.itertuples():
    try: ev=json.loads(r.evaluations) if isinstance(r.evaluations,str) else []
    except Exception: continue
    struct={}   # target -> url
    per={}      # target -> labels
    glob={}
    for e in ev:
        m=e.get('metric'); v=e.get('value'); t=e.get('target')
        if m=='boltz2_structure_prediction' and isinstance(v,dict) and v.get('url'):
            struct[t]=v['url']
        elif m in ('binding','binding_strength','expressed','kd','expression-yield'):
            per.setdefault(t,{})[m]=v
        elif m in ('design_class','esmfold_plddt','proteinmpnn_score'):
            glob[m]=v
    for t,u in struct.items():
        lab=dict(per.get(t,{}))
        d=dict(id=r.id,name=r.name,author=r.author,designMethod=r.designMethod,
               seq=r.sequence,url=u,target=t)
        d.update(glob); d.update(lab); rows.append(d)
D=pd.DataFrame(rows)
D.to_csv('/data/yash/tmp/claude-1000/-data-yash-comp/dcc873cb-83b0-4a0a-b1ac-c9a359584756/scratchpad/labels.csv',index=False)
print(len(D), "structure rows;", D.binding.notna().sum(), "labeled")
print(D[D.binding.notna()].binding.value_counts().to_dict())
print(D[D.binding.notna()].binding_strength.value_counts(dropna=False).to_dict())

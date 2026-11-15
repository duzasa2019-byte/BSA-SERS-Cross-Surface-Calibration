import numpy as np,pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
P=Path(__file__).resolve().parents[1] / 'results' / 'chemical'
D=pd.read_csv(str(Path(__file__).resolve().parents[1] / 'data' / '07052025 - BSA-thiol_dataset_corrected.csv'))
wn=np.arange(401,1800); X=D[[str(i) for i in wn]].to_numpy(float)
# Per-spectrum SNV only; does not pool information across samples
Z=(X-X.mean(axis=1,keepdims=True))/np.maximum(X.std(axis=1,keepdims=True),1e-12)
bands={'Phe_1003':(990,1015),'Tyr_830_850':(820,860),'Amide_III':(1230,1300),'CH_bending':(1430,1480),'Amide_I':(1600,1700)}
rows=[]
for (surface,conc),ix in D.groupby(['label','conc']).groups.items():
 idx=np.array(list(ix)); r={'surface':surface.replace('-BSA',''),'concentration_mg_ml':conc,'n_spectra':len(idx)}
 for name,(lo,hi) in bands.items():
  m=(wn>=lo)&(wn<=hi)
  r[name+'_median_SNV']=float(np.median(Z[idx][:,m].mean(axis=1)))
  r[name+'_median_raw']=float(np.median(X[idx][:,m].mean(axis=1)))
 rows.append(r)
cell=pd.DataFrame(rows).sort_values(['surface','concentration_mg_ml']);cell.to_csv(P/'band_cell_medians.csv',index=False)
tr=[]
for surface,g in cell[cell.concentration_mg_ml>0].groupby('surface'):
 g=g.sort_values('concentration_mg_ml');xx=np.log10(g.concentration_mg_ml.to_numpy())
 for name in bands:
  for scale in ['SNV','raw']:
   yy=g[name+'_median_'+scale].to_numpy(); rho,p=spearmanr(xx,yy)
   blank=cell[(cell.surface==surface)&(cell.concentration_mg_ml==0)][name+'_median_'+scale].iloc[0]
   tr.append({'surface':surface,'band':name,'scale':scale,'spearman_rho_5_levels':rho,'descriptive_p_not_for_inference':p,'highest_minus_blank':yy[-1]-blank,'highest_minus_lowest_positive':yy[-1]-yy[0]})
trend=pd.DataFrame(tr);trend.to_csv(P/'band_concentration_trends.csv',index=False)
# Between-surface chemical shift diagnostic: difference in 5-level trends of band medians
r=trend[trend.scale=='SNV'].pivot(index='band',columns='surface',values='spearman_rho_5_levels')
r.to_csv(P/'cross_surface_trend_matrix.csv')
print(r.round(3).to_string());print('N',len(D),'cells',len(cell),'trend rows',len(trend))

import numpy as np,pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
P=Path(__file__).resolve().parents[1] / 'results' / 'chemical'
D=pd.read_csv(str(Path(__file__).resolve().parents[1] / 'data' / '07052025 - BSA-thiol_dataset_corrected.csv')); pred=pd.read_csv(str(Path(__file__).resolve().parents[1] / 'results' / 'nested' / 'nested_predictions.csv'))
wn=np.arange(401,1800);X=D[[str(i) for i in wn]].to_numpy(float);Z=(X-X.mean(1)[:,None])/np.maximum(X.std(1)[:,None],1e-12)
B={'Phe_1003':(990,1015),'Tyr_830_850':(820,860),'Amide_III':(1230,1300),'CH_bending':(1430,1480),'Amide_I':(1600,1700)}
assert np.allclose(pred.actual.to_numpy(),np.log10(D.loc[pred.row_id,'conc'].to_numpy()))
rows=[]
for width in [0,10,20]:
 for scale,A in [('SNV',Z),('raw',X)]:
  for band,(lo,hi) in B.items():
   mask=(wn>=lo-width)&(wn<=hi+width)
   bandval=A[:,mask].mean(axis=1)
   for (surface,model,conc),g in pred.groupby(['held_surface','model','actual']):
    ids=g.row_id.to_numpy(int);err=np.abs(g.pred-g.actual).to_numpy()
    # Robust band median in exactly same evaluated spectra; fixed physical units per dataset
    rows.append({'surface':surface,'model':model,'log10_conc':conc,'band':band,'scale':scale,'window_extension_cm1':width,'n':len(ids),'median_band':np.median(bandval[ids]),'mae':np.mean(err),'bias':np.mean(g.pred-g.actual),'rmse':np.sqrt(np.mean((g.pred-g.actual)**2))})
cell=pd.DataFrame(rows);cell.to_csv(P/'band_error_cells.csv',index=False)
res=[]
for (model,band,scale,width),g in cell.groupby(['model','band','scale','window_extension_cm1']):
 # center each surface to reduce spurious surface offsets; 20 concentration-surface cells, not independent replicates
 x=g.median_band-g.groupby('surface').median_band.transform('mean')
 y=g.mae-g.groupby('surface').mae.transform('mean')
 rho=spearmanr(x,y).statistic
 within=[]
 for surface,gg in g.groupby('surface'):
  within.append(spearmanr(gg.median_band,gg.mae).statistic)
 res.append({'model':model,'band':band,'scale':scale,'window_extension_cm1':width,'rho_surface_centered':rho,'median_abs_within_surface_rho':np.median(np.abs(within)),'within_surface_rhos':';'.join(f'{v:.3f}' for v in within),'n_surface_concentration_cells':len(g)})
out=pd.DataFrame(res);out.to_csv(P/'band_error_associations.csv',index=False)
print('cells',len(cell),'associations',len(out))
print(out[(out.scale=='SNV')&(out.window_extension_cm1==0)].sort_values(['model','band']).to_string(index=False))
print('range across widths',out.groupby(['model','band','scale']).rho_surface_centered.agg(['min','max']).round(3).to_string())

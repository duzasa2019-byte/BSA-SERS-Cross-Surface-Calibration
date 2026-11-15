import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import pandas as pd,numpy as np,json
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from pathlib import Path
O=Path(__file__).resolve().parents[1] / 'results' / 'nested';O.mkdir(exist_ok=True)
df=pd.read_csv(str(Path(__file__).resolve().parents[1] / 'data' / '07052025 - BSA-thiol_dataset_corrected.csv'));surface=df.label.str.replace('-BSA','',regex=False).to_numpy();conc=df.conc.to_numpy();X=df[[str(i) for i in range(401,1800)]].to_numpy(float);Xp=(X-X.mean(1,keepdims=True))/np.maximum(X.std(1,keepdims=True),1e-12)
surfaces=sorted(np.unique(surface)); counts=df.groupby(['label','conc']).size().reset_index(name='n');counts.to_csv(O/'counts.csv',index=False)
# 0 is not a valid log concentration. Evaluate zero vs positive separately; descriptive domain transfer.
zero=[]
for held in surfaces:
 tr=surface!=held;te=~tr
 est=make_pipeline(StandardScaler(),PCA(n_components=15,svd_solver='randomized',random_state=42),LogisticRegression(max_iter=500,class_weight='balanced'))
 est.fit(Xp[tr],(conc[tr]>0).astype(int));prob=est.predict_proba(Xp[te])[:,1];truth=(conc[te]>0).astype(int)
 zero.append(dict(held_surface=held,n=int(te.sum()),n_zero=int(sum(~truth.astype(bool))),n_positive=int(sum(truth)),auroc=float(roc_auc_score(truth,prob)) if len(set(truth))==2 else np.nan,average_precision=float(average_precision_score(truth,prob)) if len(set(truth))==2 else np.nan,balanced_accuracy=float(np.mean([(prob[truth==k]>=0.5).mean() if k==1 else (prob[truth==k]<0.5).mean() for k in [0,1]])) if len(set(truth))==2 else np.nan))
pd.DataFrame(zero).to_csv(O/'zero_detection_scores.csv',index=False)
# nested domain-only tuning: 3 source surfaces for inner leave-one-surface-out; no held-out labels used in selection
sel=conc>0;XX=Xp[sel];yy=np.log10(conc[sel]);ss=surface[sel];ii=np.where(sel)[0]
candidates=[('PLS',2),('PLS',4),('PLS',8),('SVR',1),('SVR',10),('RF',80)]
def build(fam,par):
 if fam=='PLS':return make_pipeline(StandardScaler(),PLSRegression(n_components=par,scale=False))
 if fam=='SVR':return make_pipeline(StandardScaler(),PCA(n_components=25,svd_solver='randomized',random_state=42),SVR(C=par,epsilon=0.1,gamma='scale'))
 return make_pipeline(StandardScaler(),PCA(n_components=25,svd_solver='randomized',random_state=42),RandomForestRegressor(n_estimators=par,min_samples_leaf=3,max_features=.7,n_jobs=1,random_state=42))
inner=[];outer=[];pred=[]
for held in surfaces:
 train=ss!=held;test=~train;source=sorted(set(ss[train]))
 for fam in ['PLS','SVR','RF']:
  losses=[]
  for family,par in candidates:
   if family!=fam:continue
   vals=[]
   for validation in source:
    tr=train&(ss!=validation);va=train&(ss==validation)
    model=build(family,par);model.fit(XX[tr],yy[tr]);yp=np.asarray(model.predict(XX[va])).reshape(-1)
    rmse=float(np.sqrt(np.mean((yp-yy[va])**2)));vals.append(rmse)
    inner.append(dict(outer_held=held,model=family,param=par,inner_validation=validation,n=int(va.sum()),rmse=rmse))
   losses.append((float(np.mean(vals)),par))
  score,chosen=min(losses,key=lambda x:(x[0],x[1]));model=build(fam,chosen);model.fit(XX[train],yy[train]);yp=np.asarray(model.predict(XX[test])).reshape(-1)
  outer.append(dict(held_surface=held,model=fam,selected_param=chosen,inner_mean_rmse=score,n=int(test.sum()),rmse=float(np.sqrt(np.mean((yp-yy[test])**2))),mae=float(np.mean(np.abs(yp-yy[test]))),bias=float(np.mean(yp-yy[test]))))
  pred.extend(dict(held_surface=held,model=fam,row_id=int(j),actual=float(a),pred=float(p)) for j,a,p in zip(ii[test],yy[test],yp))
  print(held,fam,chosen,round(outer[-1]['rmse'],3),flush=True)
pd.DataFrame(inner).to_csv(O/'nested_inner_scores.csv',index=False);pd.DataFrame(outer).to_csv(O/'nested_outer_scores.csv',index=False);pd.DataFrame(pred).to_csv(O/'nested_predictions.csv',index=False)
print('ZERO',pd.DataFrame(zero).to_string(index=False));print('OUTER',pd.DataFrame(outer).to_string(index=False))

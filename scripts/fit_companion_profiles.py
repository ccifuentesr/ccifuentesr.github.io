"""Illustrative two-Gaussian fits to the two strongest resolved CCF minima.
A/B denote deeper/shallower minima, not established orbital identities.
Single-minimum epochs are retained without inventing a decomposition.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'media/hidden-companions/ccf.json')
args=parser.parse_args()
p=args.input;d=json.loads(p.read_text())
peaks=[p['rv_kms'] for e in d['epochs'] if e['accepted'] for p in e.get('peaks',[])[:2]]
low=float(np.floor((min(peaks)-30)/10)*10);high=float(np.ceil((max(peaks)+30)/10)*10)
pivot=(low+high)/2
def model(x,c,s,a,mu,wa,b,nu,wb):
 return c+s*(x-pivot)-a*np.exp(-.5*((x-mu)/wa)**2)-b*np.exp(-.5*((x-nu)/wb)**2)
for e in d['epochs']:
 if not e['accepted']:continue
 e['display_components']=None
 peaks=e.get('peaks',[])[:2]
 if len(peaks)<2:continue
 x=np.array(e['velocity']);y=np.array(e['ccf']);keep=(x>=low)&(x<=high)
 first,second=peaks
 guess=[.99,0,max(.015,1-first['ccf_value']),first['rv_kms'],4,max(.015,1-second['ccf_value']),second['rv_kms'],4]
 lo=[.85,-.002,.005,first['rv_kms']-4,1,.005,second['rv_kms']-4,1]
 hi=[1.05,.002,.4,first['rv_kms']+4,20,.4,second['rv_kms']+4,20]
 try:pars,_=curve_fit(model,x[keep],y[keep],p0=guess,bounds=(lo,hi),maxfev=20000)
 except (ValueError,RuntimeError):continue
 c,s,a,mu,wa,b,nu,wb=pars
 # Do not present two distinct components when the fitted centres collapse.
 if abs(mu-nu)<max(wa,wb):continue
 e['display_components']={'method':f'Two Gaussian dips + linear baseline, {low:g}..{high:g} km/s; illustrative fit',
 'a':{'depth':float(a),'velocity':float(mu),'sigma':float(wa)},'b':{'depth':float(b),'velocity':float(nu),'sigma':float(wb)},'baseline':float(c),'slope':float(s),'pivot':pivot}
 print(e['date'],round(mu,2),round(nu,2),round(wa,2),round(wb,2))
d['display_range']=[low,high]
p.write_text(json.dumps(d,indent=2)+'\n')
p.with_name('ccf-data.js').write_text('window.hiddenCompanionsData = '+p.read_text().strip()+';\n')

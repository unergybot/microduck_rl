import csv
import itertools
import json
from pathlib import Path
import sys

first, second = map(Path,sys.argv[1:3])
a=json.loads((first/'report.json').read_text())
b=json.loads((second/'report.json').read_text())
am={x['id']:x for x in json.loads((first/'manifest.json').read_text())['artifacts']}
bm={x['id']:x for x in json.loads((second/'manifest.json').read_text())['artifacts']}
old={(c['scenario'],c['seed']):c for c in a['cases'] if c['candidateId']=='web-delay08'}
new={(c['scenario'],c['seed']):c for c in b['cases'] if c['candidateId']=='flag-retain'}
result={'oldCandidate':'web-delay08','repeatCandidate':'flag-retain','caseCount':len(old),'sameCaseKeys':old.keys()==new.keys(),'statusReasonMatches':0,'exactMetricMatches':0,'exactCommonCsvMatches':0,'comparedCsvRows':0,'comparedCsvCells':0,'differences':[]}
for key,x in old.items():
 y=new[key]
 result['statusReasonMatches']+=int((x['status'],x['reason'])==(y['status'],y['reason']))
 result['exactMetricMatches']+=int(x['metrics']==y['metrics'])
 if x['metrics']!=y['metrics']:
  result['differences'].append({'case':key,'metricDifferences':{k:[v,y['metrics'].get(k)] for k,v in x['metrics'].items() if v!=y['metrics'].get(k)}})
 same=True
 with (first/am[x['csvArtifactId']]['path']).open() as f1,(second/bm[y['csvArtifactId']]['path']).open() as f2:
  r1,r2=csv.DictReader(f1),csv.DictReader(f2)
  assert set(r1.fieldnames)<=set(r2.fieldnames)
  for index,(row1,row2) in enumerate(itertools.zip_longest(r1,r2)):
   if row1 is None or row2 is None:
    same=False
    result['differences'].append({'case':key,'rowCountMismatchAt':index})
    break
   result['comparedCsvRows']+=1
   result['comparedCsvCells']+=len(r1.fieldnames)
   diff={k:[row1[k],row2[k]] for k in r1.fieldnames if row1[k]!=row2[k]}
   if diff:
    same=False
    if len(result['differences'])<10:
     result['differences'].append({'case':key,'row':index,'cells':diff})
 result['exactCommonCsvMatches']+=int(same)
result['deterministic']=result['sameCaseKeys'] and all(result[k]==len(old) for k in ('statusReasonMatches','exactMetricMatches','exactCommonCsvMatches'))
print(json.dumps(result,indent=2))

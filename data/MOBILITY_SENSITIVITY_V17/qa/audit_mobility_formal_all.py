import csv, math
from collections import defaultdict
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'code'/'Results_Paper2_Revision'/'MOBILITY_SENSITIVITY_FORMAL_V1'
sigmas=(15,35,70)
scenarios=('compute_hotspot','link_hotspot','coupled_hotspot')
algorithms=('FullResched','BudgetedLNS')
def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
all_run=[]; all_slot=[]
for sigma in sigmas:
    d=base/f'sigma_{sigma:03d}m'
    rr=rows(d/'Paper2_Revision_RunLevel.csv'); ss=rows(d/'Paper2_Revision_SlotLevel.csv')
    assert len(rr)==180,(sigma,'run rows',len(rr))
    assert len(ss)==36000,(sigma,'slot rows',len(ss))
    all_run+=rr; all_slot+=ss

def rk(x):return (int(float(x['MobilitySigmaMeters'])),x['Scenario'],int(x['Run']),x['Algorithm'])
def sk(x):return rk(x)+(int(x['Slot']),)
rkeys=[rk(x) for x in all_run]; skeys=[sk(x) for x in all_slot]
assert len(set(rkeys))==len(rkeys),'duplicate run keys'
assert len(set(skeys))==len(skeys),'duplicate slot keys'
expected=set((sig,sc,r,a) for sig in sigmas for sc in scenarios for r in range(1,31) for a in algorithms)
assert set(rkeys)==expected, f'run key gaps/extras {expected ^ set(rkeys)}'
by=defaultdict(list)
for x in all_slot:by[rk(x)].append(x)
for x in all_run:
    k=rk(x); sig,sc,run,alg=k; g=by[k]
    assert x['Mode']=='MOBILITY_FORMAL' and int(x['N'])==180 and int(x['Slots'])==200 and int(x['OnsetSlot'])==1
    assert float(x['BudgetFraction'])==0.05 and float(x['Seed'])==float(g[0]['Seed'])
    assert len(g)==200 and {int(y['Slot']) for y in g}==set(range(1,201)),(k,'slots')
    assert all(int(y['MobilitySigmaMeters'])==sig and y['Scenario']==sc and y['Algorithm']==alg and int(y['Run'])==run for y in g)
    assert all(int(y['AdmittedTasks'])+int(y['RejectedTasks'])==180 for y in g),(k,'admission')
    assert all(abs(float(y['ResourceViolationRate']))<1e-12 for y in g),(k,'resource violation')
    must_finite=('Satisfaction','Migrations','MigrationMB','AdmittedTasks','RejectedTasks','RuntimeSec','ResourceViolationRate','EvaluatorCalls','RepairCalls','LocalSearchInvocations','InnerUpdates','SlotPipelineSeconds')
    assert all(math.isfinite(float(y[f])) for y in g for f in must_finite),(k,'nonfinite')
    assert all(y['HotspotActive'].strip().lower() in ('true','1') for y in g)
# all run metrics recoverable from the slot table
checks={
'MeanSatisfaction':lambda g:sum(float(z['Satisfaction']) for z in g)/len(g),
'PostOnsetMeanSatisfaction':lambda g:sum(float(z['Satisfaction']) for z in g)/len(g),
'OnsetWindowMeanSatisfaction':lambda g:sum(float(z['Satisfaction']) for z in g if int(z['Slot'])<21)/20,
'TotalMigrations':lambda g:sum(float(z['Migrations']) for z in g),
'PostOnsetMigrations':lambda g:sum(float(z['Migrations']) for z in g),
'MigrationMB':lambda g:sum(float(z['MigrationMB']) for z in g),
'PostOnsetMigrationMB':lambda g:sum(float(z['MigrationMB']) for z in g),
'AdmissionTransitions':lambda g:sum(float(z['AdmissionTransitions']) for z in g),
'RejectionTransitions':lambda g:sum(float(z['RejectionTransitions']) for z in g),
'TotalPlacementStateTransitions':lambda g:sum(float(z['TotalPlacementStateTransitions']) for z in g),
'MeanAdmittedTasks':lambda g:sum(float(z['AdmittedTasks']) for z in g)/len(g),
'MeanRejectedTasks':lambda g:sum(float(z['RejectedTasks']) for z in g)/len(g),
'PostMeanAdmittedTasks':lambda g:sum(float(z['AdmittedTasks']) for z in g)/len(g),
'PostMeanRejectedTasks':lambda g:sum(float(z['RejectedTasks']) for z in g)/len(g),
'RuntimeSec':lambda g:sum(float(z['RuntimeSec']) for z in g),
'RepairOverrides':lambda g:sum(1.0 if z['RepairOverride'].strip().lower() in ('true','1') else 0.0 for z in g),
'MaxBudgetExcess':lambda g:max(float(z['BudgetExcess']) for z in g),
'EvaluatorCalls':lambda g:sum(float(z['EvaluatorCalls']) for z in g),
'RepairCalls':lambda g:sum(float(z['RepairCalls']) for z in g),
'LocalSearchInvocations':lambda g:sum(float(z['LocalSearchInvocations']) for z in g),
'InnerUpdates':lambda g:sum(float(z['InnerUpdates']) for z in g),
'EvaluatorSeconds':lambda g:sum(float(z['EvaluatorSeconds']) for z in g),
'RepairSeconds':lambda g:sum(float(z['RepairSeconds']) for z in g),
'LocalSearchSeconds':lambda g:sum(float(z['LocalSearchSeconds']) for z in g),
'SlotPipelineSeconds':lambda g:sum(float(z['SlotPipelineSeconds']) for z in g),}
worst=(0,None)
for x in all_run:
    g=by[rk(x)]
    assert not math.isfinite(float(x['PreOnsetMeanSatisfaction'])),'pre-onset should be NA at onset=1'
    for f,fn in checks.items():
        calc=fn(g); delta=abs(float(x[f])-calc)
        if delta>worst[0]:worst=(delta,(rk(x),f,float(x[f]),calc))
        assert delta<=1e-7*max(1,abs(calc)),(rk(x),f,x[f],calc)
# paired seeds must be consistent across methods and sigma levels
seed_map={}
for x in all_run:
    k=(x['Scenario'],int(x['Run']))
    seed_map.setdefault(k,set()).add(int(float(x['Seed'])))
assert all(len(v)==1 for v in seed_map.values()),'paired seed differs across methods/sigmas'
print('FORMAL FULL AUDIT PASS')
for sig in sigmas:print(f'sigma={sig}m: 180 run rows; 36,000 slot rows; 90 paired scenario-seed cells')
print(f'combined: 540 run rows; 108,000 slot rows; unique keys and all 200-slot sequences PASS')
print('admission conservation PASS; resource violations=0; required metrics finite PASS')
print(f'run-to-slot reconciliation PASS; max absolute delta={worst[0]:.3g}')
print('paired seed consistency across methods and all three sigma levels PASS')


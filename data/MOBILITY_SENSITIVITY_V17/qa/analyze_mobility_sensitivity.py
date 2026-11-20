import csv, math, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'code'/'Results_Paper2_Revision'/'MOBILITY_SENSITIVITY_FORMAL_V1'
OUT=Path(__file__).resolve().parent
SIGMAS=(15,35,70)
SCENARIOS=('compute_hotspot','link_hotspot','coupled_hotspot')
METHODS=('FullResched','BudgetedLNS')
B=20000

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def boot_ci(values, seed):
    rng=random.Random(seed); n=len(values); means=[]
    for _ in range(B):
        total=0.0
        for _ in range(n): total += values[rng.randrange(n)]
        means.append(total/n)
    means.sort()
    def q(p):
        x=(len(means)-1)*p; lo=int(x); hi=min(lo+1,len(means)-1); frac=x-lo
        return means[lo]*(1-frac)+means[hi]*frac
    return q(.025),q(.975)

data={}
for sigma in SIGMAS:
    f=RESULTS/f'sigma_{sigma:03d}m'/'Paper2_Revision_RunLevel.csv'
    rows=read_csv(f)
    assert len(rows)==180,(sigma,len(rows))
    for r in rows:
        key=(sigma,r['Scenario'],int(r['Run']),r['Algorithm'])
        assert key not in data,key
        data[key]={'sat':float(r['MeanSatisfaction']),'moves':float(r['TotalMigrations']),
                   'seed':int(float(r['Seed']))}
assert len(data)==540
for scenario in SCENARIOS:
    for run in range(1,31):
        seeds={data[(s,scenario,run,m)]['seed'] for s in SIGMAS for m in METHODS}
        assert len(seeds)==1,(scenario,run,seeds)

pairs={}
for sigma in SIGMAS:
    for scenario in SCENARIOS:
        arr=[]
        for run in range(1,31):
            f=data[(sigma,scenario,run,'FullResched')]
            b=data[(sigma,scenario,run,'BudgetedLNS')]
            # MeanSatisfaction is already stored on the 0--100 percentage scale;
            # its difference is therefore directly in percentage points.
            arr.append((run,b['sat']-f['sat'],b['moves']-f['moves'],f['moves'],b['moves']))
        pairs[(sigma,scenario)]=arr

summary=[]
for sigma in SIGMAS:
    for si,scenario in enumerate(SCENARIOS):
        arr=pairs[(sigma,scenario)]
        sat=[x[1] for x in arr]; mdiff=[x[2] for x in arr]
        fmean=sum(x[3] for x in arr)/len(arr); bmean=sum(x[4] for x in arr)/len(arr)
        assert fmean>0,(sigma,scenario,'zero FullResched migrations')
        satci=boot_ci(sat,20261120+sigma*100+si)
        mdci=boot_ci(mdiff,20262120+sigma*100+si)
        ratios=[]; rng=random.Random(20263120+sigma*100+si)
        for _ in range(B):
            sample=[arr[rng.randrange(len(arr))] for _ in arr]
            fs=sum(x[3] for x in sample); bs=sum(x[4] for x in sample)
            if fs>0: ratios.append(100.0*(1.0-bs/fs))
        ratios.sort()
        def qv(p):
            x=(len(ratios)-1)*p; lo=int(x); hi=min(lo+1,len(ratios)-1); z=x-lo
            return ratios[lo]*(1-z)+ratios[hi]*z
        assert len(ratios)>=int(.99*B),(sigma,scenario,'unstable ratio denominator')
        better=sum(v>1e-12 for v in sat); tied=sum(abs(v)<=1e-12 for v in sat)
        summary.append({
            'SigmaMeters':sigma,'Scenario':scenario,'PairedSeeds':len(arr),
            'FullMeanSatisfaction':sum(data[(sigma,scenario,r,'FullResched')]['sat'] for r in range(1,31))/30,
            'BudgetedLNSMeanSatisfaction':sum(data[(sigma,scenario,r,'BudgetedLNS')]['sat'] for r in range(1,31))/30,
            'LNSMinusFullSatisfaction_pp':sum(sat)/30,'SatisfactionDeltaCI95Low_pp':satci[0],
            'SatisfactionDeltaCI95High_pp':satci[1],'LNSBetterSeeds':better,'TiedSeeds':tied,
            'FullMeanMigrations_200slots':fmean,'LNSMeanMigrations_200slots':bmean,
            'LNSMinusFullMigrations':sum(mdiff)/30,'MigrationDeltaCI95Low':mdci[0],
            'MigrationDeltaCI95High':mdci[1],
            'MigrationReductionPct_of_group_means':100*(1-bmean/fmean),
            'MigrationReductionCI95Low_pct':qv(.025),'MigrationReductionCI95High_pct':qv(.975),
            'BootstrapResamples':B})

interactions=[]
for scenario in SCENARIOS:
    for sigma in (15,70):
        sat_change=[]; move_change=[]
        for run in range(1,31):
            x=next(z for z in pairs[(sigma,scenario)] if z[0]==run)
            base=next(z for z in pairs[(35,scenario)] if z[0]==run)
            sat_change.append(x[1]-base[1]); move_change.append(x[2]-base[2])
        si=SCENARIOS.index(scenario)
        satci=boot_ci(sat_change,20264120+sigma*100+si)
        mvci=boot_ci(move_change,20265120+sigma*100+si)
        interactions.append({'Scenario':scenario,'ComparedSigmaMeters':sigma,'ReferenceSigmaMeters':35,
            'PairedSeeds':30,'ChangeInSatisfactionAdvantage_pp':sum(sat_change)/30,
            'ChangeInSatisfactionAdvantageCI95Low_pp':satci[0],
            'ChangeInSatisfactionAdvantageCI95High_pp':satci[1],
            'ChangeInMigrationDifference':sum(move_change)/30,
            'ChangeInMigrationDifferenceCI95Low':mvci[0],
            'ChangeInMigrationDifferenceCI95High':mvci[1],
            'BootstrapResamples':B})

for name,rows in [('mobility_sensitivity_effects.csv',summary),('mobility_sensitivity_interactions.csv',interactions)]:
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

report=['# Mobility-intensity sensitivity: paired formal analysis','',
        'Data: 30 paired run seeds per scenario and sigma; 200 slots per run; paired methods share scenario/seed inputs.',
        'Primary contrasts: BudgetedLNS minus FullResched satisfaction (percentage points) and total migrations per 200-slot run.',
        f'Intervals: paired percentile bootstrap, {B:,} resamples, fixed seeds recorded in the analysis script. No p-values or universal-trend claim are made.', '',
        '| Sigma (m) | Scenario | Δ satisfaction, pp (95% CI) | Δ migrations (95% CI) | Migration reduction from group means | Better satisfaction seeds / 30 |',
        '|---:|---|---:|---:|---:|---:|']
for x in summary:
    report.append(f"| {x['SigmaMeters']} | {x['Scenario']} | {x['LNSMinusFullSatisfaction_pp']:.2f} ({x['SatisfactionDeltaCI95Low_pp']:.2f}, {x['SatisfactionDeltaCI95High_pp']:.2f}) | {x['LNSMinusFullMigrations']:.2f} ({x['MigrationDeltaCI95Low']:.2f}, {x['MigrationDeltaCI95High']:.2f}) | {x['MigrationReductionPct_of_group_means']:.2f}% ({x['MigrationReductionCI95Low_pct']:.2f}, {x['MigrationReductionCI95High_pct']:.2f}) | {x['LNSBetterSeeds']} / 30 |")
report+=['','## Change relative to the 35 m setting','',
         '| Scenario | Sigma vs 35 m | Change in satisfaction advantage, pp (95% CI) | Change in migration difference (95% CI) |',
         '|---|---:|---:|---:|']
for x in interactions:
    report.append(f"| {x['Scenario']} | {x['ComparedSigmaMeters']} m | {x['ChangeInSatisfactionAdvantage_pp']:.2f} ({x['ChangeInSatisfactionAdvantageCI95Low_pp']:.2f}, {x['ChangeInSatisfactionAdvantageCI95High_pp']:.2f}) | {x['ChangeInMigrationDifference']:.2f} ({x['ChangeInMigrationDifferenceCI95Low']:.2f}, {x['ChangeInMigrationDifferenceCI95High']:.2f}) |")
report+=['','## Scope note','',
         'These are simulator-level results for the tested task mix, five-node environment, three hotspot scenarios, and independent per-slot Gaussian location jitter. The analysis tests mobility intensity only; it does not test task-composition or hotspot-severity sensitivity and does not establish behavior under physical user trajectories.']
(OUT/'mobility_sensitivity_analysis.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('PAIRED ANALYSIS COMPLETE')
print(f'groups={len(summary)} interaction_contrasts={len(interactions)} paired_bootstrap_resamples={B:,}')
print(f'outputs: {OUT / "mobility_sensitivity_analysis.md"}')
print(f'outputs: {OUT / "mobility_sensitivity_effects.csv"}')
print(f'outputs: {OUT / "mobility_sensitivity_interactions.csv"}')


import csv, math
from collections import defaultdict, Counter
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "Results_Paper2_Revision" / "MOBILITY_SENSITIVITY_SMOKE_V1"
sigmas = (15, 35, 70)
scenarios = ("compute_hotspot", "link_hotspot", "coupled_hotspot")
algorithms = ("FullResched", "BudgetedLNS")
expected_runs = set((s, sc, r, a) for s in sigmas for sc in scenarios for r in (1,2) for a in algorithms)
run_rows, slot_rows = [], []
for sigma in sigmas:
    folder = base / f"sigma_{sigma:03d}m"
    for fname, target in (("Paper2_Revision_RunLevel.csv", run_rows), ("Paper2_Revision_SlotLevel.csv", slot_rows)):
        path = folder / fname
        assert path.is_file(), f"Missing {path}"
        with path.open(newline="", encoding="utf-8-sig") as f:
            target.extend(dict(x) for x in csv.DictReader(f))
assert len(run_rows) == 36, f"run rows={len(run_rows)} expected 36"
assert len(slot_rows) == 1080, f"slot rows={len(slot_rows)} expected 1080"

def key_run(x): return (int(float(x["MobilitySigmaMeters"])), x["Scenario"], int(x["Run"]), x["Algorithm"])
def key_slot(x): return key_run(x) + (int(x["Slot"]),)
run_keys = [key_run(x) for x in run_rows]
slot_keys = [key_slot(x) for x in slot_rows]
assert len(set(run_keys)) == len(run_keys), "duplicate run key"
assert len(set(slot_keys)) == len(slot_keys), "duplicate slot key"
assert set(run_keys) == expected_runs, f"missing/extra run cells: {expected_runs ^ set(run_keys)}"
by_run = defaultdict(list)
for x in slot_rows:
    by_run[key_run(x)].append(x)

for x in run_rows:
    k = key_run(x); sigma, scenario, run, algo = k
    assert float(x["N"]) == 180 and int(x["Slots"]) == 30 and int(x["OnsetSlot"]) == 1
    assert abs(float(x["BudgetFraction"]) - .05) < 1e-12
    assert x["Mode"] == "MOBILITY_SMOKE"
    group = by_run[k]
    assert len(group) == 30 and {int(z["Slot"]) for z in group} == set(range(1,31)), f"bad slots {k}"
    assert all(int(z["N"]) == 180 and int(z["MobilitySigmaMeters"]) == sigma for z in group)
    assert all(z["Scenario"] == scenario and z["Algorithm"] == algo and int(z["Run"]) == run for z in group)
    assert all(int(z["AdmittedTasks"])+int(z["RejectedTasks"]) == 180 for z in group), f"admission accounting {k}"
    assert all(abs(float(z["ResourceViolationRate"])) < 1e-12 for z in group), f"resource violation {k}"
    assert all(math.isfinite(float(z[field])) for z in group for field in ("Satisfaction","Migrations","MigrationMB","AdmittedTasks","RejectedTasks","RuntimeSec","ResourceViolationRate","EvaluatorCalls","RepairCalls","LocalSearchInvocations","InnerUpdates","SlotPipelineSeconds")), f"nonfinite required metric {k}"
    assert int(float(x["Seed"])) == int(float(group[0]["Seed"]))
    assert all(int(float(z["Seed"])) == int(float(x["Seed"])) for z in group)

# Every run-level quantity that is recoverable from slot rows must reconcile.
checks = {
    "MeanSatisfaction": lambda g: sum(float(z["Satisfaction"]) for z in g)/len(g),
    "PostOnsetMeanSatisfaction": lambda g: sum(float(z["Satisfaction"]) for z in g if int(z["Slot"]) >= 1)/len(g),
    "OnsetWindowMeanSatisfaction": lambda g: sum(float(z["Satisfaction"]) for z in g if int(z["Slot"]) < 21)/20,
    "TotalMigrations": lambda g: sum(float(z["Migrations"]) for z in g),
    "PostOnsetMigrations": lambda g: sum(float(z["Migrations"]) for z in g),
    "MigrationMB": lambda g: sum(float(z["MigrationMB"]) for z in g),
    "PostOnsetMigrationMB": lambda g: sum(float(z["MigrationMB"]) for z in g),
    "AdmissionTransitions": lambda g: sum(float(z["AdmissionTransitions"]) for z in g),
    "RejectionTransitions": lambda g: sum(float(z["RejectionTransitions"]) for z in g),
    "TotalPlacementStateTransitions": lambda g: sum(float(z["TotalPlacementStateTransitions"]) for z in g),
    "MeanAdmittedTasks": lambda g: sum(float(z["AdmittedTasks"]) for z in g)/len(g),
    "MeanRejectedTasks": lambda g: sum(float(z["RejectedTasks"]) for z in g)/len(g),
    "PostMeanAdmittedTasks": lambda g: sum(float(z["AdmittedTasks"]) for z in g)/len(g),
    "PostMeanRejectedTasks": lambda g: sum(float(z["RejectedTasks"]) for z in g)/len(g),
    "RuntimeSec": lambda g: sum(float(z["RuntimeSec"]) for z in g),
    "RepairOverrides": lambda g: sum(1.0 if z["RepairOverride"].strip().lower() in ("true","1") else 0.0 for z in g),
    "MaxBudgetExcess": lambda g: max(float(z["BudgetExcess"]) for z in g),
    "EvaluatorCalls": lambda g: sum(float(z["EvaluatorCalls"]) for z in g),
    "RepairCalls": lambda g: sum(float(z["RepairCalls"]) for z in g),
    "LocalSearchInvocations": lambda g: sum(float(z["LocalSearchInvocations"]) for z in g),
    "InnerUpdates": lambda g: sum(float(z["InnerUpdates"]) for z in g),
    "EvaluatorSeconds": lambda g: sum(float(z["EvaluatorSeconds"]) for z in g),
    "RepairSeconds": lambda g: sum(float(z["RepairSeconds"]) for z in g),
    "LocalSearchSeconds": lambda g: sum(float(z["LocalSearchSeconds"]) for z in g),
    "SlotPipelineSeconds": lambda g: sum(float(z["SlotPipelineSeconds"]) for z in g),
}
max_delta = (0.0, None)
for x in run_rows:
    g = by_run[key_run(x)]
    for field, calc in checks.items():
        delta = abs(float(x[field]) - calc(g))
        if delta > max_delta[0]: max_delta = (delta, (key_run(x), field, x[field], calc(g)))
        assert delta <= 1e-7 * max(1.0, abs(calc(g))), f"summary mismatch {key_run(x)} {field}: {x[field]} vs {calc(g)}"
    assert not math.isfinite(float(x["PreOnsetMeanSatisfaction"])), "pre-onset mean should be unavailable for onset slot 1"

seed_by_cell = {(int(float(x["MobilitySigmaMeters"])), x["Scenario"], int(x["Run"])): int(float(x["Seed"])) for x in run_rows}
assert len(seed_by_cell) == 18
for sigma in sigmas:
    for scenario in scenarios:
        for run in (1,2):
            assert len({int(float(x["Seed"])) for x in run_rows if int(float(x["MobilitySigmaMeters"])) == sigma and x["Scenario"] == scenario and int(x["Run"]) == run}) == 1

print("SMOKE INDEPENDENT AUDIT PASS")
print(f"run_rows={len(run_rows)} slot_rows={len(slot_rows)} unique_run_keys={len(set(run_keys))} unique_slot_keys={len(set(slot_keys))}")
print("sigma_m=15,35,70; scenarios=3; seeds=2; methods=2; slots/run=30")
print("admission conservation=PASS; resource violations=0; finite required metrics=PASS")
print(f"run-to-slot reconciliation=PASS; max absolute delta={max_delta[0]:.3g}")
print("The reported pre-onset metric is NaN because hotspot onset is slot 1, as prespecified.")


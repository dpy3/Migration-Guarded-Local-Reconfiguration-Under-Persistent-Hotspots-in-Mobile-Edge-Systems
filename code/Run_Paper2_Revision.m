function Run_Paper2_Revision(mode, run_filter, scenario_filter, shard_id, budget_fraction)
% Versioned Paper 2 experiments: main, hotspot onset, component ablation,
% and budget sensitivity. Results never reuse the legacy Paper 2 checkpoint.
if nargin < 1 || isempty(mode), mode = 'smoke'; end
cfg = local_config(mode);
if nargin >= 2 && ~isempty(run_filter), cfg.runs = run_filter; end
if nargin >= 3 && ~isempty(scenario_filter), cfg.scenarios = cellstr(string(scenario_filter)); end
if nargin >= 5 && ~isempty(budget_fraction), cfg.budget_fraction = budget_fraction; end
cfg = finalize_config(cfg);

root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'Algorithms'), fullfile(root, 'Environment'), ...
    fullfile(root, 'Utils'), fullfile(root, 'Data'), fullfile(root, 'HIVE'));
result_root = fullfile(root, 'Results_Paper2_Revision', cfg.result_name);
if nargin >= 4 && ~isempty(shard_id)
    safe_id = regexprep(char(string(shard_id)), '[^A-Za-z0-9_-]', '_');
    result_root = fullfile(result_root, 'SHARDS', safe_id);
end
if ~exist(result_root, 'dir'), mkdir(result_root); end
checkpoint = fullfile(result_root, 'Paper2_Revision_Checkpoint.mat');
run_output = fullfile(result_root, 'Paper2_Revision_RunLevel.csv');
slot_output = fullfile(result_root, 'Paper2_Revision_SlotLevel.csv');

run_rows = repmat(base_run_row(), 0, 1);
slot_rows = repmat(base_slot_row(), 0, 1);
if exist(checkpoint, 'file') == 2
    saved = load(checkpoint, 'run_rows', 'slot_rows', 'cfg_saved');
    assert(isfield(saved, 'cfg_saved') && strcmp(saved.cfg_saved.signature, cfg.signature), ...
        'Paper2:CheckpointMismatch', 'Checkpoint protocol does not match this run.');
    run_rows = saved.run_rows;
    slot_rows = saved.slot_rows;
end

fprintf('Paper2 revision %s: runs=%s slots=%d onset=%d algorithms=%s\n', ...
    upper(cfg.mode), mat2str(cfg.runs), cfg.slots, cfg.onset_slot, strjoin(cfg.algorithms, ','));
for s = 1:numel(cfg.scenarios)
    scenario = cfg.scenarios{s};
    for run = cfg.runs
        if complete_run(run_rows, scenario, run, cfg.algorithms), continue; end
        scenario_id = find(strcmp(cfg.all_scenarios, scenario), 1);
        seed = cfg.seed_base + 1000 * scenario_id + run;
        rng(seed, 'twister');
        [Task0, Fog0, Thing0, DNN_Data] = generate_tier1_env_hetero(cfg.N, 5);
        [run_rows, slot_rows] = remove_partial(run_rows, slot_rows, scenario, run);
        previous = cell(1, numel(cfg.algorithms));
        hive_state = struct('task_age', zeros(cfg.N, 1), 'backlog_proxy', 0);
        accum = repmat(accumulator(), 1, numel(cfg.algorithms));

        for slot = 1:cfg.slots
            [Task, Fog, Thing, hotspot_active] = revision_state( ...
                Task0, Fog0, Thing0, scenario, slot, seed, cfg.onset_slot);
            Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
            Pre.Violation_Mode = 'composite';
            for a = 1:numel(cfg.algorithms)
                alg = cfg.algorithms{a};
                algorithm_id = find(strcmp(cfg.algorithms, alg), 1);
                rng(seed + 100000 * slot + 100 * algorithm_id, 'twister');
                t0 = tic;
                [theta, hive_out, solver_info] = solve_algorithm(alg, Task, Fog, Thing, ...
                    DNN_Data, Pre, previous{a}, hive_state, cfg);
                elapsed = toc(t0);
                theta = Perform_Safe_Harbor_Repair(theta, Pre, Fog, Task, DNN_Data);
                [sat, ~, breakdown] = calculate_metrics_v2(theta, Task, Fog, Thing, DNN_Data, Pre);
                assert(breakdown.resource_viol_rate == 0, 'Paper2:ResourceViolation', ...
                    '%s %s run=%d slot=%d has a resource violation.', scenario, alg, run, slot);

                if slot > 1
                    [moves, moved_mb, admission_transitions, rejection_transitions, state_transitions] = ...
                        migration_cost(theta, previous{a}, Task, DNN_Data);
                else
                    moves = 0; moved_mb = 0; admission_transitions = 0;
                    rejection_transitions = 0; state_transitions = 0;
                end
                admitted = sum(any(theta > 0, 2));
                rejected = cfg.N - admitted;
                backlog = backlog_proxy(theta, Pre);
                [guard_enabled, move_budget, repair_override, budget_excess] = ...
                    budget_diagnostics(solver_info, moves);

                accum(a) = update_accumulator(accum(a), sat, moves, moved_mb, ...
                    admission_transitions, rejection_transitions, state_transitions, admitted, ...
                    rejected, elapsed, backlog, slot, cfg, repair_override, budget_excess, solver_info);
                slot_rows(end + 1) = make_slot_row(cfg, scenario, run, seed, alg, slot, ...
                    hotspot_active, sat, moves, moved_mb, admission_transitions, ...
                    rejection_transitions, state_transitions, admitted, rejected, elapsed, ...
                    guard_enabled, move_budget, repair_override, budget_excess, breakdown); %#ok<AGROW>
                previous{a} = theta;
                if strcmp(alg, 'HIVE')
                    hive_state = update_hive_state(Task, hive_out, accum(a).BacklogSum);
                end
            end
        end

        for a = 1:numel(cfg.algorithms)
            run_rows(end + 1) = make_run_row(cfg, scenario, run, seed, cfg.algorithms{a}, accum(a)); %#ok<AGROW>
        end
        cfg_saved = cfg; %#ok<NASGU>
        save(checkpoint, 'run_rows', 'slot_rows', 'cfg_saved', '-v7.3');
        fprintf('%s run=%d complete\n', scenario, run);
    end
end
writetable(struct2table(run_rows), run_output);
writetable(struct2table(slot_rows), slot_output);
validate_results(run_rows, slot_rows, cfg);
fprintf('PAPER2 REVISION PASS: %s\n', result_root);
end

function cfg = local_config(mode)
cfg.mode = lower(char(mode));
cfg.protocol_version = 'paper2_revision_v3_transition_audit_20260906';
cfg.seed_base = 2026073100;
cfg.N = 180;
cfg.all_scenarios = {'compute_hotspot','link_hotspot','coupled_hotspot'};
cfg.scenarios = cfg.all_scenarios;
cfg.budget_fraction = 0.05;
switch cfg.mode
    case 'smoke'
        cfg.runs = 1:2; cfg.slots = 30; cfg.onset_slot = 11;
        cfg.algorithms = {'FullResched','HIVE','BigMECGreedy','BudgetedLNS'};
        cfg.result_name = 'SMOKE_V3';
    case 'formal'
        cfg.runs = 1:30; cfg.slots = 200; cfg.onset_slot = 1;
        cfg.algorithms = {'FullResched','HIVE','BigMECGreedy','BudgetedLNS'};
        cfg.result_name = 'FORMAL_V2';
    case 'onset'
        cfg.runs = 1:30; cfg.slots = 200; cfg.onset_slot = 51;
        cfg.algorithms = {'FullResched','HIVE','BigMECGreedy','BudgetedLNS'};
        cfg.result_name = 'ONSET_V2';
    case 'ablation'
        cfg.runs = 1:30; cfg.slots = 200; cfg.onset_slot = 51;
        cfg.algorithms = {'BudgetedLNS','LNSColdStart','LNSGlobalPool','LNSNoGuard'};
        cfg.result_name = 'ABLATION_V2';
    case 'sensitivity'
        cfg.runs = 1:30; cfg.slots = 200; cfg.onset_slot = 1;
        cfg.algorithms = {'BudgetedLNS'};
        cfg.result_name = 'SENSITIVITY_V2';
    otherwise
        error('Paper2:UnknownMode', 'Unknown mode: %s', mode);
end
end

function cfg = finalize_config(cfg)
assert(all(ismember(cfg.scenarios, cfg.all_scenarios)), 'Paper2:UnknownScenario', ...
    'Scenario filter contains an unknown scenario.');
assert(all(cfg.runs >= 1 & cfg.runs <= 30), 'Paper2:InvalidRun', 'Runs must be in 1:30.');
assert(cfg.budget_fraction > 0 && cfg.budget_fraction <= 1, 'Paper2:InvalidBudget', ...
    'Budget fraction must be in (0,1].');
if strcmp(cfg.mode, 'sensitivity')
    cfg.result_name = fullfile(cfg.result_name, sprintf('B%03d', round(1000 * cfg.budget_fraction)));
end
cfg.signature = sprintf('%s|%s|N%d|T%d|O%d|B%.6f|%s', cfg.protocol_version, ...
    cfg.mode, cfg.N, cfg.slots, cfg.onset_slot, cfg.budget_fraction, strjoin(cfg.algorithms, '-'));
end

function [theta, hive_out, info] = solve_algorithm(alg, Task, Fog, Thing, DNN_Data, Pre, previous, hive_state, cfg)
hive_out = struct(); info = struct();
switch alg
    case 'FullResched'
        theta = MPC_Rollout_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, struct());
    case 'HIVE'
        [theta, hive_out] = HIVE_Controller(Task, Fog, Thing, DNN_Data, Pre, struct(), hive_state);
    case 'BigMECGreedy'
        [theta, info] = BigMEC_Greedy_Adapter_Binary(Task, Fog, Thing, DNN_Data, Pre, previous);
    case {'BudgetedLNS','LNSColdStart','LNSGlobalPool','LNSNoGuard'}
        options = struct('budget_fraction', cfg.budget_fraction, 'candidate_multiplier', 3, ...
            'state_preserving', true, 'pressure_localized', true, ...
            'full_candidate_pool', false, 'enforce_budget_guard', true);
        if strcmp(alg, 'LNSColdStart'), options.state_preserving = false; end
        if strcmp(alg, 'LNSGlobalPool')
            options.pressure_localized = false;
            options.full_candidate_pool = true;
        end
        if strcmp(alg, 'LNSNoGuard'), options.enforce_budget_guard = false; end
        [theta, info] = Budgeted_LNS_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, previous, options);
    otherwise
        error('Paper2:UnknownAlgorithm', 'Unknown algorithm: %s', alg);
end
end

function [Task, Fog, Thing, active] = revision_state(Task0, Fog0, Thing0, scenario, slot, seed, onset_slot)
Task = Task0; Fog = Fog0; Thing = Thing0;
phase = 2 * pi * (slot - 1) / 20;
Task(:, 3) = max(0.12, Task0(:, 3) .* (0.93 + 0.07 * sin(phase + Task0(:, 1) / 17)));
rng(seed + 500000 + slot, 'twister');
Thing(:, 1:2) = min(1000, max(0, Thing0(:, 1:2) + 35 * randn(size(Thing0, 1), 2)));
active = slot >= onset_slot;
if ~active, return; end
switch scenario
    case 'compute_hotspot'
        Fog(1, [4, 7]) = 0.55 * Fog0(1, [4, 7]);
    case 'link_hotspot'
        Fog(2, 6) = 0.35 * Fog0(2, 6);
    case 'coupled_hotspot'
        Fog(1, [4, 7]) = 0.60 * Fog0(1, [4, 7]);
        Fog(2, 6) = 0.40 * Fog0(2, 6);
        Fog(2, 8) = 0.70 * Fog0(2, 8);
end
end

function a = accumulator()
a = struct('SatSum',0,'Migrations',0,'MigrationMB',0,'AdmissionTransitions',0, ...
    'RejectionTransitions',0,'StateTransitions',0,'Runtime',0,'BacklogSum',0, ...
    'Admitted',0,'Rejected',0,'PreSat',0,'PreCount',0,'PostSat',0,'PostCount',0, ...
    'OnsetSat',0,'OnsetCount',0,'PostMigrations',0,'PostMigrationMB',0, ...
    'PostAdmitted',0,'PostRejected',0,'RepairOverrides',0,'MaxBudgetExcess',0, ...
    'CandidateEvaluations',0,'AcceptedMoves',0,'BudgetActiveSlots',0);
end

function a = update_accumulator(a, sat, moves, moved_mb, admission_transitions, rejection_transitions, state_transitions, admitted, rejected, elapsed, backlog, slot, cfg, repair_override, budget_excess, info)
a.SatSum = a.SatSum + sat; a.Migrations = a.Migrations + moves;
a.MigrationMB = a.MigrationMB + moved_mb; a.Runtime = a.Runtime + elapsed;
a.AdmissionTransitions = a.AdmissionTransitions + admission_transitions;
a.RejectionTransitions = a.RejectionTransitions + rejection_transitions;
a.StateTransitions = a.StateTransitions + state_transitions;
a.BacklogSum = a.BacklogSum + backlog;
a.Admitted = a.Admitted + admitted; a.Rejected = a.Rejected + rejected;
a.RepairOverrides = a.RepairOverrides + double(repair_override);
a.MaxBudgetExcess = max(a.MaxBudgetExcess, budget_excess);
a.CandidateEvaluations = a.CandidateEvaluations + get_field(info, 'CandidateEvaluations', 0);
a.AcceptedMoves = a.AcceptedMoves + get_field(info, 'AcceptedMoves', 0);
a.BudgetActiveSlots = a.BudgetActiveSlots + double(get_field(info, 'MoveBudget', inf) <= get_field(info, 'Migrations', 0));
if slot < cfg.onset_slot
    a.PreSat = a.PreSat + sat; a.PreCount = a.PreCount + 1;
else
    a.PostSat = a.PostSat + sat; a.PostCount = a.PostCount + 1;
    a.PostMigrations = a.PostMigrations + moves; a.PostMigrationMB = a.PostMigrationMB + moved_mb;
    a.PostAdmitted = a.PostAdmitted + admitted; a.PostRejected = a.PostRejected + rejected;
    if slot < cfg.onset_slot + 20
        a.OnsetSat = a.OnsetSat + sat; a.OnsetCount = a.OnsetCount + 1;
    end
end
end

function row = base_run_row()
row = struct('Protocol',"",'Mode',"",'Scenario',"",'Run',0,'Seed',0,'N',0,'Slots',0, ...
    'OnsetSlot',0,'Algorithm',"",'BudgetFraction',0,'MeanSatisfaction',0, ...
    'PreOnsetMeanSatisfaction',NaN,'PostOnsetMeanSatisfaction',0,'OnsetWindowMeanSatisfaction',0, ...
    'TotalMigrations',0,'PostOnsetMigrations',0,'MigrationMB',0,'PostOnsetMigrationMB',0, ...
    'AdmissionTransitions',0,'RejectionTransitions',0,'TotalPlacementStateTransitions',0, ...
    'MeanAdmittedTasks',0,'MeanRejectedTasks',0,'PostMeanAdmittedTasks',0,'PostMeanRejectedTasks',0, ...
    'RuntimeSec',0,'RepairOverrides',0,'MaxBudgetExcess',0,'CandidateEvaluations',0, ...
    'AcceptedMoves',0,'BudgetActiveSlots',0);
end

function row = make_run_row(cfg, scenario, run, seed, alg, a)
row = base_run_row(); row.Protocol = string(cfg.protocol_version); row.Mode = upper(string(cfg.mode));
row.Scenario = string(scenario); row.Run = run; row.Seed = seed; row.N = cfg.N;
row.Slots = cfg.slots; row.OnsetSlot = cfg.onset_slot; row.Algorithm = string(alg);
row.BudgetFraction = cfg.budget_fraction; row.MeanSatisfaction = a.SatSum / cfg.slots;
if a.PreCount > 0, row.PreOnsetMeanSatisfaction = a.PreSat / a.PreCount; end
row.PostOnsetMeanSatisfaction = a.PostSat / max(a.PostCount, 1);
row.OnsetWindowMeanSatisfaction = a.OnsetSat / max(a.OnsetCount, 1);
row.TotalMigrations = a.Migrations; row.PostOnsetMigrations = a.PostMigrations;
row.MigrationMB = a.MigrationMB; row.PostOnsetMigrationMB = a.PostMigrationMB;
row.AdmissionTransitions = a.AdmissionTransitions;
row.RejectionTransitions = a.RejectionTransitions;
row.TotalPlacementStateTransitions = a.StateTransitions;
row.MeanAdmittedTasks = a.Admitted / cfg.slots; row.MeanRejectedTasks = a.Rejected / cfg.slots;
row.PostMeanAdmittedTasks = a.PostAdmitted / max(a.PostCount, 1);
row.PostMeanRejectedTasks = a.PostRejected / max(a.PostCount, 1);
row.RuntimeSec = a.Runtime; row.RepairOverrides = a.RepairOverrides;
row.MaxBudgetExcess = a.MaxBudgetExcess; row.CandidateEvaluations = a.CandidateEvaluations;
row.AcceptedMoves = a.AcceptedMoves; row.BudgetActiveSlots = a.BudgetActiveSlots;
end

function row = base_slot_row()
row = struct('Protocol',"",'Mode',"",'Scenario',"",'Run',0,'Seed',0,'N',0, ...
    'Slot',0,'OnsetSlot',0,'HotspotActive',false,'Algorithm',"",'BudgetFraction',0, ...
    'Satisfaction',0,'Migrations',0,'MigrationMB',0,'AdmissionTransitions',0, ...
    'RejectionTransitions',0,'TotalPlacementStateTransitions',0,'AdmittedTasks',0,'RejectedTasks',0, ...
    'RuntimeSec',0,'BudgetGuardEnabled',false,'MoveBudget',NaN,'RepairOverride',false, ...
    'BudgetExcess',0,'ResourceViolationRate',0);
end

function row = make_slot_row(cfg, scenario, run, seed, alg, slot, active, sat, moves, moved_mb, admission_transitions, rejection_transitions, state_transitions, admitted, rejected, elapsed, guard_enabled, move_budget, repair_override, budget_excess, breakdown)
row = base_slot_row(); row.Protocol = string(cfg.protocol_version); row.Mode = upper(string(cfg.mode));
row.Scenario = string(scenario); row.Run = run; row.Seed = seed; row.N = cfg.N;
row.Slot = slot; row.OnsetSlot = cfg.onset_slot; row.HotspotActive = active;
row.Algorithm = string(alg); row.BudgetFraction = cfg.budget_fraction;
row.Satisfaction = sat; row.Migrations = moves; row.MigrationMB = moved_mb;
row.AdmissionTransitions = admission_transitions;
row.RejectionTransitions = rejection_transitions;
row.TotalPlacementStateTransitions = state_transitions;
row.AdmittedTasks = admitted; row.RejectedTasks = rejected; row.RuntimeSec = elapsed;
row.BudgetGuardEnabled = guard_enabled; row.MoveBudget = move_budget;
row.RepairOverride = repair_override; row.BudgetExcess = budget_excess;
row.ResourceViolationRate = breakdown.resource_viol_rate;
end

function [enabled, budget, override, excess] = budget_diagnostics(info, moves)
enabled = logical(get_field(info, 'BudgetGuardEnabled', false));
budget = get_field(info, 'MoveBudget', NaN);
if isfinite(budget)
    excess = max(moves - budget, 0);
    override = logical(enabled && excess > 0);
else
    excess = 0;
    override = false;
end
end

function [moves, moved_mb, admission_transitions, rejection_transitions, state_transitions] = migration_cost(theta, previous, Task, DNN_Data)
[~, now] = max(theta, [], 2); [~, before] = max(previous, [], 2);
now(sum(theta, 2) == 0) = 0; before(sum(previous, 2) == 0) = 0;
changed = now > 0 & before > 0 & now ~= before;
admission_transitions = sum(before == 0 & now > 0);
rejection_transitions = sum(before > 0 & now == 0);
state_transitions = sum(now ~= before);
moves = sum(changed); moved_mb = 0;
for i = find(changed)'
    type = Task(i, 2);
    if iscell(DNN_Data) && type <= numel(DNN_Data) && isfield(DNN_Data{type}, 'data')
        moved_mb = moved_mb + sum(DNN_Data{type}.data);
    end
end
end

function state = update_hive_state(Task, hive_out, backlog)
state = struct('task_age', zeros(size(Task, 1), 1), 'backlog_proxy', backlog);
if ~isfield(hive_out, 'breakdown') || ~isfield(hive_out.breakdown, 'detail'), return; end
d = hive_out.breakdown.detail;
if isfield(d, 'satisfied_vec'), state.task_age(d.satisfied_vec(:) <= 0) = max(Task(:, 3)); end
end

function value = backlog_proxy(theta, Pre)
load = sum(theta .* Pre.Comp, 1);
value = sum(load .^ 2);
end

function value = get_field(s, name, default)
value = default; if isstruct(s) && isfield(s, name), value = s.(name); end
end

function tf = complete_run(rows, scenario, run, algorithms)
if isempty(rows), tf = false; return; end
mask = string({rows.Scenario}) == string(scenario) & [rows.Run] == run;
tf = isequal(sort(string({rows(mask).Algorithm})), sort(string(algorithms)));
end

function [runs, slots] = remove_partial(runs, slots, scenario, run)
if ~isempty(runs)
    mask = string({runs.Scenario}) == string(scenario) & [runs.Run] == run; runs(mask) = [];
end
if ~isempty(slots)
    mask = string({slots.Scenario}) == string(scenario) & [slots.Run] == run; slots(mask) = [];
end
end

function validate_results(run_rows, slot_rows, cfg)
runs = struct2table(run_rows); slots = struct2table(slot_rows);
expected_runs = numel(cfg.scenarios) * numel(cfg.runs) * numel(cfg.algorithms);
expected_slots = expected_runs * cfg.slots;
assert(height(runs) == expected_runs, 'Paper2:IncompleteRuns', 'Expected %d run rows, found %d.', expected_runs, height(runs));
assert(height(slots) == expected_slots, 'Paper2:IncompleteSlots', 'Expected %d slot rows, found %d.', expected_slots, height(slots));
assert(numel(unique(strcat(runs.Scenario,"|",string(runs.Run),"|",runs.Algorithm))) == height(runs), 'Paper2:DuplicateRun', 'Duplicate run keys.');
assert(numel(unique(strcat(slots.Scenario,"|",string(slots.Run),"|",slots.Algorithm,"|",string(slots.Slot)))) == height(slots), 'Paper2:DuplicateSlot', 'Duplicate slot keys.');
assert(all(slots.AdmittedTasks + slots.RejectedTasks == cfg.N), 'Paper2:AdmissionAccounting', 'Admitted + rejected must equal N for every slot.');
assert(all(abs(slots.ResourceViolationRate) < 1e-12), 'Paper2:ResourceViolation', 'A slot has nonzero resource violation.');
assert(all(abs(runs.MeanAdmittedTasks + runs.MeanRejectedTasks - cfg.N) < 1e-9), 'Paper2:RunAdmissionAccounting', 'Run-level admitted + rejected must equal N.');
end

function [theta, info] = Budgeted_LNS_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, theta_previous, options)
% Local destroy-and-repair search with a guard on repaired local trials.
% A final feasibility repair may exceed the guard; BudgetRespected records it.
[N, M] = size(Pre.Comp);
if nargin < 6, theta_previous = []; end
if nargin < 7 || isempty(options), options = struct(); end
if ~isfield(options, 'budget_fraction'), options.budget_fraction = 0.05; end
if ~isfield(options, 'candidate_multiplier'), options.candidate_multiplier = 3; end
if ~isfield(options, 'state_preserving'), options.state_preserving = true; end
if ~isfield(options, 'pressure_localized'), options.pressure_localized = true; end
if ~isfield(options, 'full_candidate_pool'), options.full_candidate_pool = false; end
if ~isfield(options, 'enforce_budget_guard'), options.enforce_budget_guard = true; end
move_budget = max(1, ceil(options.budget_fraction * N));

has_previous = ~isempty(theta_previous) && isequal(size(theta_previous), [N, M]);
if has_previous
    theta_reference = Perform_Safe_Harbor_Repair(theta_previous, Pre, Fog, Task, DNN_Data);
else
    theta_reference = MPC_Rollout_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, struct());
    theta_reference = Perform_Safe_Harbor_Repair(theta_reference, Pre, Fog, Task, DNN_Data);
end
if options.state_preserving && has_previous
    theta = theta_reference;
else
    theta = MPC_Rollout_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, struct());
    theta = Perform_Safe_Harbor_Repair(theta, Pre, Fog, Task, DNN_Data);
end
[sat_current, ~, b_current] = calculate_metrics_v2(theta, Task, Fog, Thing, DNN_Data, Pre);

pressure = node_pressure(theta, Pre, Fog, Task, DNN_Data);
[~, pressured_nodes] = sort(pressure, 'descend');
pressured_nodes = pressured_nodes(1:min(2, M));
[~, assigned] = max(theta, [], 2);
admitted = sum(theta, 2) > 0;
assigned(~admitted) = 0;
if options.pressure_localized
    pool = find(ismember(assigned, pressured_nodes));
else
    pool = find(admitted);
end
risk = zeros(numel(pool), 1);
for k = 1:numel(pool)
    i = pool(k); f = assigned(i);
    risk(k) = (Pre.Comm(i, f) + Pre.Comp(i, f)) / max(Task(i, 3), 1e-9);
end
[~, order] = sort(risk, 'descend');
if options.full_candidate_pool
    pool = pool(order);
else
    pool = pool(order(1:min(numel(order), options.candidate_multiplier * move_budget)));
end

accepted = 0;
evaluations = 0;
for k = 1:numel(pool)
    if options.enforce_budget_guard && accepted >= move_budget, break; end
    i = pool(k);
    current_node = find(theta(i, :) > 0, 1);
    best_theta = theta;
    best_sat = sat_current;
    best_migrations = migration_count(theta, theta_reference);
    best_latency = pick_latency(b_current);
    for f = 1:M
        if isequal(f, current_node), continue; end
        trial = theta;
        trial(i, :) = 0;
        trial(i, f) = 1;
        trial = Perform_Safe_Harbor_Repair(trial, Pre, Fog, Task, DNN_Data);
        migrations = migration_count(trial, theta_reference);
        if options.enforce_budget_guard && migrations > move_budget, continue; end
        [sat, ~, breakdown] = calculate_metrics_v2(trial, Task, Fog, Thing, DNN_Data, Pre);
        evaluations = evaluations + 1;
        latency = pick_latency(breakdown);
        if better_candidate(sat, migrations, latency, best_sat, best_migrations, best_latency)
            best_theta = trial;
            best_sat = sat;
            best_migrations = migrations;
            best_latency = latency;
        end
    end
    if ~isequal(best_theta, theta)
        theta = best_theta;
        sat_current = best_sat;
        [~, ~, b_current] = calculate_metrics_v2(theta, Task, Fog, Thing, DNN_Data, Pre);
        accepted = migration_count(theta, theta_reference);
    end
end

theta = Perform_Safe_Harbor_Repair(theta, Pre, Fog, Task, DNN_Data);
final_migrations = migration_count(theta, theta_reference);
info = struct('MoveBudget', move_budget, ...
    'Migrations', final_migrations, ...
    'CandidateEvaluations', evaluations, 'AcceptedMoves', accepted, ...
    'BudgetGuardEnabled', logical(options.enforce_budget_guard), ...
    'BudgetRespected', final_migrations <= move_budget, ...
    'RepairOverride', logical(options.enforce_budget_guard && final_migrations > move_budget), ...
    'StatePreserving', logical(options.state_preserving), ...
    'PressureLocalized', logical(options.pressure_localized), ...
    'FullCandidatePool', logical(options.full_candidate_pool));
end

function tf = better_candidate(sat, migrations, latency, best_sat, best_migrations, best_latency)
tol = 1e-10;
tf = sat > best_sat + tol || ...
    (abs(sat - best_sat) <= tol && migrations < best_migrations) || ...
    (abs(sat - best_sat) <= tol && migrations == best_migrations && latency < best_latency - tol);
end

function count = migration_count(theta, reference)
[~, now] = max(theta, [], 2);
[~, before] = max(reference, [], 2);
now(sum(theta, 2) == 0) = 0;
before(sum(reference, 2) == 0) = 0;
count = sum(now > 0 & before > 0 & now ~= before);
end

function pressure = node_pressure(theta, Pre, Fog, Task, DNN_Data)
M = size(theta, 2);
rho = sum(theta .* Pre.Comp, 1) / max(mean(Task(:, 3)), 1e-9);
mem = zeros(1, M);
for i = 1:size(theta, 1)
    f = find(theta(i, :) > 0, 1);
    if ~isempty(f), mem(f) = mem(f) + task_memory(Task(i, 2), DNN_Data); end
end
if size(Fog, 2) >= 8, caps = Fog(:, 8)'; else, caps = Fog(:, end)'; end
pressure = rho + mem ./ max(caps, 1);
end

function value = task_memory(task_type, DNN_Data)
if iscell(DNN_Data) && task_type <= numel(DNN_Data) && isfield(DNN_Data{task_type}, 'data')
    value = sum(DNN_Data{task_type}.data);
elseif isnumeric(DNN_Data)
    map = [3, 2, 1]; row = map(min(max(round(task_type), 1), 3));
    value = sum(DNN_Data(row, 1:min(2, size(DNN_Data, 2))));
else
    value = 100;
end
end

function value = pick_latency(b)
value = inf;
if isfield(b, 'latency_admitted_all') && isfinite(b.latency_admitted_all)
    value = b.latency_admitted_all;
elseif isfield(b, 'latency_admitted_timely') && isfinite(b.latency_admitted_timely)
    value = b.latency_admitted_timely;
end
end

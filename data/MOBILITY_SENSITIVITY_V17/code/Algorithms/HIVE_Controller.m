function [theta_out, hive_out] = HIVE_Controller(Task, Fog, Thing, DNN_Data, Pre_In, cfg, runtime_state)
% HIVE_Controller
% Minimal first-pass implementation of HIVE:
% 1) build service state
% 2) discover fragility cells
% 3) estimate propagation
% 4) plan hierarchical interventions
% 5) realize sparse micro-edits
% 6) run a short counterfactual stabilization check
% 7) apply shared repair

    N = size(Task, 1); %#ok<NASGU>
    M = size(Fog, 1); %#ok<NASGU>
    if nargin < 5 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end
    if nargin < 6 || isempty(cfg)
        cfg = local_default_cfg(size(Fog, 1));
    else
        cfg = local_merge_cfg(cfg, size(Fog, 1));
    end
    if nargin < 7 || isempty(runtime_state)
        runtime_state = struct();
    end

    state = local_build_service_state(Task, Fog, Thing, DNN_Data, Pre, cfg, runtime_state);
    cells = Discover_Fragility_Cells(state, cfg);
    cells = Estimate_Fragility_Propagation(cells, state, cfg);
    plan = Plan_Hierarchical_Intervention(cells, state, cfg);
    candidates = Realize_Sparse_Micro_Edits(plan, state, cfg);
    best = Counterfactual_Stabilization_Check(candidates, state, cfg);

    theta_candidate = best.theta_candidate;
    theta_out = Perform_Safe_Harbor_Repair(theta_candidate, Pre, Fog, Task, DNN_Data);
    [sat, viol, breakdown] = calculate_metrics_v2(theta_out, Task, Fog, Thing, DNN_Data, Pre);

    hive_out = struct();
    if isempty(cells)
        cell_priority_scores = [];
        cell_fragility_scores = [];
        cell_propagation_scores = [];
    else
        cell_priority_scores = [cells.priority_score];
        cell_fragility_scores = [cells.local_fragility];
        cell_propagation_scores = [cells.propagation_score];
    end

    hive_out.summary_metrics = struct( ...
        'satisfaction', sat, ...
        'violation', viol, ...
        'latency', local_pick_latency(breakdown), ...
        'global_fragility_potential', sum(cell_priority_scores), ...
        'mean_cell_fragility', local_safe_mean(cell_fragility_scores), ...
        'max_cell_fragility', local_safe_max(cell_fragility_scores), ...
        'mean_propagation_score', local_safe_mean(cell_propagation_scores), ...
        'max_propagation_score', local_safe_max(cell_propagation_scores), ...
        'num_cells', numel(cells), ...
        'num_interventions', numel(plan.target_cell_ids), ...
        'num_edits', best.num_edits, ...
        'edit_cost', best.edit_cost, ...
        'selected_program_mode', string(local_safe_field(best, 'program_mode', 'Base')), ...
        'selected_program_variant', string(local_safe_field(best, 'program_variant', 'BaseGreedy')), ...
        'selected_program_score', local_safe_field(best, 'rank_score', inf));
    hive_out.cell_log = cells;
    hive_out.intervention_log = local_build_intervention_log(cells, plan, best);
    hive_out.micro_edit_log = best.edit_list;
    hive_out.selected_candidate = best;
    hive_out.slot_fragility_potential = sum(cell_priority_scores);
    hive_out.num_cells = numel(cells);
    hive_out.num_interventions = numel(plan.target_cell_ids);
    hive_out.num_edits = best.num_edits;
    hive_out.breakdown = breakdown;
end

function value = local_safe_field(s, field_name, default_value)
    if isstruct(s) && isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    else
        value = default_value;
    end
end

function cfg = local_default_cfg(M)
    cfg = struct();
    cfg.HIVE_MAX_CELLS = 5;
    cfg.HIVE_MAX_TARGET_CELLS = 2;
    cfg.HIVE_MAX_EDITS = 4;
    cfg.HIVE_QUEUE_W = 1.0;
    cfg.HIVE_MEM_W = 1.0;
    cfg.HIVE_DEADLINE_W = 1.2;
    cfg.HIVE_LOAD_W = 0.8;
    cfg.HIVE_SKEW_W = 0.6;
    cfg.HIVE_PROP_SHARED_NODE_W = 1.0;
    cfg.HIVE_PROP_TASK_W = 0.8;
    cfg.HIVE_PROP_LINK_W = 0.8;
    cfg.HIVE_BACKLOG_W = 1.0;
    cfg.HIVE_FRAGILITY_W = 1.0;
    cfg.HIVE_EDIT_COST_W = 0.2;
    cfg.HIVE_NODE_TOPK = max(1, min(M, 2));
    cfg.HIVE_ENABLE_PROPAGATION = true;
    cfg.HIVE_ENABLE_COUNTERFACTUAL = true;
    cfg.HIVE_ENABLE_EDITS = true;
    cfg.HIVE_FORCE_SINGLE_MODE = '';
    cfg.HIVE_COUNTERFACTUAL_HORIZON = 3;
    cfg.HIVE_COUNTERFACTUAL_LOAD_W = 1.2;
    cfg.HIVE_COUNTERFACTUAL_DEADLINE_W = 1.5;
    cfg.HIVE_COUNTERFACTUAL_MEMORY_W = 1.3;
    cfg.HIVE_COUNTERFACTUAL_PROP_W = 1.4;
    cfg.HIVE_COUNTERFACTUAL_CONCENTRATION_W = 1.0;
    cfg.HIVE_EDIT_TRIGGER_GAIN = 0.85;
    cfg.HIVE_AGGRESSIVE_EDIT_BONUS = 2;
    cfg.HIVE_STRESS_MODE = 'none';
    cfg.HIVE_PROGRAM_MIN_EDITS = 1;
    cfg.HIVE_PROGRAM_MIX_BONUS = 1;
    cfg.HIVE_BASE_RETREAT_PENALTY = 0.0;
    cfg.HIVE_FORCE_NONBASE_UNDER_FRAGILITY = true;
    cfg.HIVE_FRAGILITY_RETREAT_GATE = 1.25;
    cfg.HIVE_MEMORY_MAX_DEST_RATIO = 0.88;
    cfg.HIVE_QUARANTINE_LAT_SLACK = 0.22;
    cfg.HIVE_PROP_CLUSTER_BONUS = 1.0;
    cfg.HIVE_PROP_SHARED_HOTSPOT_BONUS = 1.0;
    cfg.HIVE_PROP_DROP_REWARD_W = 1.4;
end

function cfg = local_merge_cfg(cfg, M)
    base = local_default_cfg(M);
    f = fieldnames(base);
    for i = 1:numel(f)
        if ~isfield(cfg, f{i})
            cfg.(f{i}) = base.(f{i});
        end
    end
end

function state = local_build_service_state(Task, Fog, Thing, DNN_Data, Pre, cfg, runtime_state)
    %#ok<INUSD>
    N = size(Task, 1);
    M = size(Fog, 1);
    mean_deadline = max(mean(Task(:, 3)), 1e-3);
    comp_min = min(Pre.Comp, [], 2);
    comm_min = min(Pre.Comm, [], 2);
    est_best_lat = comp_min + comm_min;
    deadline_slack = Task(:, 3) - est_best_lat;

    task_mem_mb = zeros(N, 1);
    for i = 1:N
        task_mem_mb(i) = local_get_task_mem_mb(Task(i, 2), DNN_Data);
    end

    node_nominal_load = sum(Pre.Comp, 1) ./ max(N, 1);
    node_mem_caps = local_get_node_memory_caps(Fog, M);
    node_queue_proxy = node_nominal_load / mean_deadline;
    node_load = node_queue_proxy;
    node_mem_margin = node_mem_caps;

    link_pressure = mean(Pre.Comm, 1);
    backlog_proxy = 0;
    task_age = zeros(N, 1);
    if isfield(runtime_state, 'task_age') && numel(runtime_state.task_age) == N
        task_age = runtime_state.task_age(:);
    end
    if isfield(runtime_state, 'backlog_proxy')
        backlog_proxy = runtime_state.backlog_proxy;
    end

    state = struct();
    state.Task = Task;
    state.Fog = Fog;
    state.Thing = Thing;
    state.DNN_Data = DNN_Data;
    state.Pre = Pre;
    state.N = N;
    state.M = M;
    state.deadline_slack = deadline_slack;
    state.task_age = task_age;
    state.task_mem_mb = task_mem_mb;
    state.node_queue_proxy = node_queue_proxy(:);
    state.node_load = node_load(:);
    state.node_mem_margin = node_mem_margin(:);
    state.link_pressure = link_pressure(:);
    state.backlog_proxy = backlog_proxy;
end

function logs = local_build_intervention_log(cells, plan, best)
    logs = struct([]);
    edit_count = numel(best.edit_list);
    for i = 1:numel(plan.target_cell_ids)
        cell_id = plan.target_cell_ids(i);
        if cell_id < 1 || cell_id > numel(cells)
            continue;
        end
        logs(i).slot_id = 1;
        logs(i).cell_id = cell_id;
        logs(i).cell_type = cells(cell_id).type;
        logs(i).intervention_type = char(plan.intervention_types{i});
        logs(i).num_micro_edits = edit_count;
        logs(i).pre_intervention_fragility = cells(cell_id).local_fragility;
        logs(i).post_intervention_fragility_est = max(0, cells(cell_id).local_fragility - 0.5);
        logs(i).pre_intervention_backlog = 0;
        logs(i).post_intervention_backlog_est = 0;
        logs(i).counterfactual_rank = best.selected_idx;
        logs(i).selected_candidate_score = best.rank_score;
    end
end

function mem_mb = local_get_task_mem_mb(t_type, DNN_Data)
    if isnumeric(DNN_Data)
        if t_type == 1
            row_idx = 3;
        elseif t_type == 2
            row_idx = 2;
        elseif t_type == 3
            row_idx = 1;
        else
            row_idx = 1;
        end
        if size(DNN_Data, 1) >= row_idx && size(DNN_Data, 2) >= 2
            mem_mb = DNN_Data(row_idx, 2) + DNN_Data(row_idx, 1);
        else
            mem_mb = 100;
        end
        return;
    end

    if iscell(DNN_Data)
        if t_type > length(DNN_Data)
            t_type = 1;
        end
        model = DNN_Data{t_type};
        if isfield(model, 'data')
            mem_mb = sum(model.data);
        else
            mem_mb = 100;
        end
        return;
    end

    if isstruct(DNN_Data)
        if t_type > numel(DNN_Data)
            t_type = 1;
        end
        model = DNN_Data(t_type);
        if isfield(model, 'data')
            mem_mb = sum(model.data);
        else
            mem_mb = 100;
        end
        return;
    end

    mem_mb = 100;
end

function mem_caps = local_get_node_memory_caps(Fog, M)
    if size(Fog, 2) >= 8
        mem_caps = Fog(:, 8)';
    else
        mem_caps = Fog(:, end)';
    end
    if numel(mem_caps) ~= M
        mem_caps = reshape(mem_caps, 1, []);
        mem_caps = mem_caps(1:min(numel(mem_caps), M));
        if numel(mem_caps) < M
            mem_caps = [mem_caps, inf(1, M - numel(mem_caps))];
        end
    end
end

function value = local_pick_latency(breakdown)
    value = NaN;
    if isstruct(breakdown)
        if isfield(breakdown, 'latency_admitted_all') && isfinite(breakdown.latency_admitted_all)
            value = breakdown.latency_admitted_all;
        elseif isfield(breakdown, 'latency_satisfied') && isfinite(breakdown.latency_satisfied)
            value = breakdown.latency_satisfied;
        elseif isfield(breakdown, 'latency_admitted_timely') && isfinite(breakdown.latency_admitted_timely)
            value = breakdown.latency_admitted_timely;
        end
    end
end

function x = local_safe_mean(vec)
    if isempty(vec)
        x = 0;
    else
        x = mean(vec);
    end
end

function x = local_safe_max(vec)
    if isempty(vec)
        x = 0;
    else
        x = max(vec);
    end
end

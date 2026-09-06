function best = Counterfactual_Stabilization_Check(candidates, state, cfg)
% Counterfactual_Stabilization_Check
% Multi-program selector over HIVE candidate families.

    blank_best = struct( ...
        'theta_candidate', zeros(state.N, state.M), ...
        'num_edits', 0, ...
        'edit_list', struct([]), ...
        'edit_cost', 0, ...
        'candidate_label', 'Empty', ...
        'program_mode', 'Empty', ...
        'program_variant', 'Empty', ...
        'program_group', 0, ...
        'program_mix', string("Empty"), ...
        'bottleneck_type', string("empty"), ...
        'intended_outcome_mode', string("empty"), ...
        'rank_score', inf, ...
        'selected_idx', 0, ...
        'selected_program_rank', 0, ...
        'counterfactual_trace', [], ...
        'program_table', table());
    if isempty(candidates)
        best = blank_best;
        return;
    end

    scores = inf(numel(candidates), 1);
    traces = cell(numel(candidates), 1);
    rows = struct('CandidateIdx', {}, 'ProgramGroup', {}, 'ProgramMode', {}, 'ProgramVariant', {}, ...
        'BottleneckType', {}, 'IntendedOutcomeMode', {}, ...
        'Score', {}, 'Satisfaction', {}, 'Violation', {}, 'Latency', {}, 'Backlog', {}, ...
        'FutureRisk', {}, 'Concentration', {}, 'MemoryPressure', {}, 'DeadlineTail', {}, ...
        'Propagation', {}, 'PropagationDrop', {}, 'NumEdits', {});
    for c = 1:numel(candidates)
        theta = candidates(c).theta_candidate;
        [sat, viol, breakdown] = calculate_metrics_v2(theta, state.Task, state.Fog, state.Thing, state.DNN_Data, state.Pre);
        latency = local_pick_latency(breakdown);
        if ~isfinite(latency)
            latency = 1e4;
        end

        backlog_proxy = local_safe_backlog_proxy(theta, state.Pre);
        imbalance_proxy = std(sum(theta .* state.Pre.Comp, 1));
        aging_proxy = mean(max(0, state.task_age - state.Task(:, 3)));
        future_risk = local_counterfactual_future_risk(theta, state, cfg);
        concentration_proxy = local_concentration_proxy(theta, state);
        mem_pressure_proxy = local_memory_pressure_proxy(theta, state);
        deadline_tail_proxy = local_deadline_tail_proxy(theta, state);
        propagation_proxy = local_propagation_proxy(theta, state);
        baseline_propagation_proxy = local_baseline_propagation_proxy(state);
        propagation_drop = max(0, baseline_propagation_proxy - propagation_proxy);
        sat_gain = sat - local_baseline_satisfaction_proxy(state);
        satisfied_gain = local_satisfied_count_gain(theta, state);

        if isfield(cfg, 'HIVE_ENABLE_COUNTERFACTUAL') && ~cfg.HIVE_ENABLE_COUNTERFACTUAL
            future_weight = 0;
            concentration_weight = 0;
            memory_weight = 0;
            deadline_weight = 0;
            propagation_weight = 0;
            propagation_reward_weight = 0;
            satisfied_gain_weight = 0;
        else
            future_weight = cfg.HIVE_FRAGILITY_W;
            concentration_weight = cfg.HIVE_COUNTERFACTUAL_CONCENTRATION_W;
            memory_weight = cfg.HIVE_COUNTERFACTUAL_MEMORY_W;
            deadline_weight = cfg.HIVE_COUNTERFACTUAL_DEADLINE_W;
            propagation_weight = cfg.HIVE_COUNTERFACTUAL_PROP_W;
            propagation_reward_weight = cfg.HIVE_PROP_DROP_REWARD_W;
            satisfied_gain_weight = local_satisfied_gain_weight(cfg, candidates(c));
        end

        reward_term = 1.2 * sat - 0.25 * max(0, state.backlog_proxy - backlog_proxy) + ...
            propagation_reward_weight * propagation_drop + ...
            1.1 * max(0, sat_gain) + ...
            satisfied_gain_weight * satisfied_gain;
        risk_term = cfg.HIVE_BACKLOG_W * backlog_proxy + ...
            future_weight * (state.backlog_proxy + future_risk) + ...
            0.35 * imbalance_proxy + ...
            0.15 * aging_proxy + ...
            concentration_weight * concentration_proxy + ...
            memory_weight * mem_pressure_proxy + ...
            deadline_weight * deadline_tail_proxy + ...
            propagation_weight * propagation_proxy + ...
            cfg.HIVE_EDIT_COST_W * max(candidates(c).edit_cost - 1, 0) + ...
            0.04 * latency + 0.25 * viol;

        base_penalty = 0;
        is_base_candidate = strcmpi(local_safe_field(candidates(c), 'program_mode', 'Base'), 'Base');
        frag_level = local_fragility_level(state);
        is_medium_prop = local_is_medium_propagation_state(state, cfg);
        if is_base_candidate
            if isfield(cfg, 'HIVE_BASE_RETREAT_PENALTY') && ~isempty(cfg.HIVE_BASE_RETREAT_PENALTY)
                base_penalty = base_penalty + cfg.HIVE_BASE_RETREAT_PENALTY;
            end
            if isfield(cfg, 'HIVE_FORCE_NONBASE_UNDER_FRAGILITY') && cfg.HIVE_FORCE_NONBASE_UNDER_FRAGILITY && ...
                    frag_level >= cfg.HIVE_FRAGILITY_RETREAT_GATE
                base_penalty = base_penalty + 8 + 2 * frag_level;
            end
            if strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation') && propagation_drop <= 0.98 * max(baseline_propagation_proxy, 1e-3)
                base_penalty = base_penalty + 6 + 2.5 * propagation_proxy;
            end
            if strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation') && candidates(c).num_edits == 0
                base_penalty = base_penalty + 4 + 2.0 * max(0, frag_level);
            end
            if is_medium_prop
                base_penalty = base_penalty + 5.0 + 1.25 * max(0, frag_level) + 0.60 * max(0, baseline_propagation_proxy - propagation_proxy);
            end
        else
            reward_term = reward_term + 0.35 * candidates(c).num_edits;
            if strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation')
                variant_name = lower(local_safe_field(candidates(c), 'program_variant', ''));
                reward_term = reward_term + local_propagation_variant_bonus(variant_name, propagation_drop, candidates(c).num_edits);
                if is_medium_prop
                    reward_term = reward_term + local_medium_propagation_variant_bonus(variant_name, propagation_drop, candidates(c).num_edits);
                end
            end
        end

        scores(c) = risk_term - reward_term + base_penalty;

        traces{c} = struct( ...
            'satisfaction', sat, ...
            'violation', viol, ...
            'latency', latency, ...
            'backlog_proxy', backlog_proxy, ...
            'imbalance_proxy', imbalance_proxy, ...
            'aging_proxy', aging_proxy, ...
            'future_risk', future_risk, ...
            'concentration_proxy', concentration_proxy, ...
            'mem_pressure_proxy', mem_pressure_proxy, ...
            'deadline_tail_proxy', deadline_tail_proxy, ...
            'propagation_proxy', propagation_proxy, ...
            'baseline_propagation_proxy', baseline_propagation_proxy, ...
            'propagation_drop', propagation_drop, ...
            'candidate_label', candidates(c).candidate_label, ...
            'program_mode', local_safe_field(candidates(c), 'program_mode', 'Base'), ...
            'program_variant', local_safe_field(candidates(c), 'program_variant', 'BaseGreedy'), ...
            'program_group', local_safe_field(candidates(c), 'program_group', 0), ...
            'bottleneck_type', local_safe_field(candidates(c), 'bottleneck_type', "base"), ...
            'intended_outcome_mode', local_safe_field(candidates(c), 'intended_outcome_mode', "baseline"));

        rows(c) = struct( ...
            'CandidateIdx', c, ...
            'ProgramGroup', local_safe_field(candidates(c), 'program_group', 0), ...
            'ProgramMode', string(local_safe_field(candidates(c), 'program_mode', 'Base')), ...
            'ProgramVariant', string(local_safe_field(candidates(c), 'program_variant', 'BaseGreedy')), ...
            'BottleneckType', string(local_safe_field(candidates(c), 'bottleneck_type', 'base')), ...
            'IntendedOutcomeMode', string(local_safe_field(candidates(c), 'intended_outcome_mode', 'baseline')), ...
            'Score', scores(c), ...
            'Satisfaction', sat, ...
            'Violation', viol, ...
            'Latency', latency, ...
            'Backlog', backlog_proxy, ...
            'FutureRisk', future_risk, ...
            'Concentration', concentration_proxy, ...
            'MemoryPressure', mem_pressure_proxy, ...
            'DeadlineTail', deadline_tail_proxy, ...
            'Propagation', propagation_proxy, ...
            'PropagationDrop', propagation_drop, ...
            'NumEdits', candidates(c).num_edits);
    end

    program_tbl = struct2table(rows);
    if strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation')
        program_tbl = sortrows(program_tbl, {'Satisfaction', 'Score', 'Latency'}, {'descend', 'ascend', 'ascend'});
    else
        program_tbl = sortrows(program_tbl, {'Score', 'Satisfaction', 'Latency'}, {'ascend', 'descend', 'ascend'});
    end
    selected_candidate_idx = program_tbl.CandidateIdx(1);
    selected_program_rank = 1;
    idx = selected_candidate_idx;
    best = candidates(idx);
    best.rank_score = scores(idx);
    best.selected_idx = idx;
    best.selected_program_rank = selected_program_rank;
    best.counterfactual_trace = traces{idx};
    best.program_table = program_tbl;
end

function weight = local_satisfied_gain_weight(cfg, candidate)
    weight = 0.75;
    if strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation')
        weight = 1.35;
    end
    intended_mode = lower(string(local_safe_field(candidate, 'intended_outcome_mode', '')));
    variant_name = lower(string(local_safe_field(candidate, 'program_variant', '')));
    if contains(intended_mode, "aggressive_relief")
        weight = weight + 1.20;
    end
    if contains(variant_name, "threshold") || contains(variant_name, "aggressive")
        weight = weight + 0.65;
    end
end

function gain = local_satisfied_count_gain(theta, state)
    assign = local_assign_vec(theta, state.N);
    current_sat = 0;
    baseline_sat = 0;
    baseline_assign = local_baseline_assign_for_propagation(state);
    for i = 1:state.N
        fi = assign(i);
        if fi > 0
            est = state.Pre.Comm(i, fi) + state.Pre.Comp(i, fi);
            current_sat = current_sat + double(est <= 1.02 * state.Task(i, 3));
        end
        fb = baseline_assign(i);
        if fb > 0
            estb = state.Pre.Comm(i, fb) + state.Pre.Comp(i, fb);
            baseline_sat = baseline_sat + double(estb <= 1.02 * state.Task(i, 3));
        end
    end
    gain = max(0, current_sat - baseline_sat) / max(state.N, 1);
end

function sat = local_baseline_satisfaction_proxy(state)
    baseline_assign = local_baseline_assign_for_propagation(state);
    sat_count = 0;
    for i = 1:state.N
        fb = baseline_assign(i);
        if fb <= 0
            continue;
        end
        est = state.Pre.Comm(i, fb) + state.Pre.Comp(i, fb);
        sat_count = sat_count + double(est <= 1.02 * state.Task(i, 3));
    end
    sat = 100 * sat_count / max(state.N, 1);
end

function bonus = local_propagation_variant_bonus(variant_name, propagation_drop, num_edits)
    bonus = 0;
    if contains(variant_name, 'clustertriadguard')
        bonus = bonus + 1.55;
    elseif contains(variant_name, 'coordinateddrain')
        bonus = bonus + 1.25;
    elseif contains(variant_name, 'clusterdecouple')
        bonus = bonus + 1.10;
    elseif contains(variant_name, 'twohopspillrelief')
        bonus = bonus + 0.95;
    elseif contains(variant_name, 'drain')
        bonus = bonus + 0.55;
    end
    bonus = bonus + 0.18 * max(num_edits - 1, 0) + 0.22 * max(propagation_drop, 0);
end

function bonus = local_medium_propagation_variant_bonus(variant_name, propagation_drop, num_edits)
    bonus = 0;
    if contains(variant_name, 'clustertriadguard')
        bonus = bonus + 2.10;
    elseif contains(variant_name, 'clusterdecouple')
        bonus = bonus + 1.60;
    elseif contains(variant_name, 'twohopspillrelief')
        bonus = bonus + 1.35;
    elseif contains(variant_name, 'coordinateddrain')
        bonus = bonus + 1.10;
    elseif contains(variant_name, 'drain')
        bonus = bonus + 0.70;
    end
    bonus = bonus + 0.20 * max(num_edits - 2, 0) + 0.18 * max(propagation_drop - 3.2, 0);
end

function tf = local_is_medium_propagation_state(state, cfg)
    tf = strcmpi(local_safe_field(cfg, 'HIVE_STRESS_MODE', 'none'), 'propagation') && ...
        isfield(state, 'N') && state.N >= 200 && state.N <= 240;
end

function frag_level = local_fragility_level(state)
    frag_level = 0;
    if isfield(state, 'backlog_proxy')
        frag_level = frag_level + state.backlog_proxy;
    end
    frag_level = frag_level + mean(max(0, -state.deadline_slack));
    if isfield(state, 'node_queue_proxy')
        frag_level = frag_level + mean(state.node_queue_proxy);
    end
end

function future_risk = local_counterfactual_future_risk(theta, state, cfg)
    node_base_time = sum(theta .* state.Pre.Comp, 1);
    normalized_load = node_base_time / max(mean(state.Task(:, 3)), 1e-3);
    load_tail = max(normalized_load) + mean(normalized_load .^ 2);
    fragility_tail = mean(max(0, -state.deadline_slack));
    propagation_tail = local_linked_spill_proxy(theta, state);
    memory_tail = local_memory_pressure_proxy(theta, state);
    horizon = max(1, cfg.HIVE_COUNTERFACTUAL_HORIZON);
    future_risk = horizon * ( ...
        cfg.HIVE_COUNTERFACTUAL_LOAD_W * load_tail + ...
        cfg.HIVE_COUNTERFACTUAL_DEADLINE_W * fragility_tail + ...
        cfg.HIVE_COUNTERFACTUAL_PROP_W * propagation_tail + ...
        cfg.HIVE_COUNTERFACTUAL_MEMORY_W * memory_tail);
end

function spill = local_linked_spill_proxy(theta, state)
    assign = local_assign_vec(theta, state.N);
    spill = local_propagation_proxy_from_assign(assign, state);
end

function backlog_proxy = local_safe_backlog_proxy(theta, Pre)
    node_base_time = sum(theta .* Pre.Comp, 1);
    backlog_proxy = sum(node_base_time .^ 2) / max(numel(node_base_time), 1);
end

function conc = local_concentration_proxy(theta, state)
    node_task_counts = sum(theta, 1);
    comp_share = sum(theta .* state.Pre.Comp, 1);
    node_load = comp_share / max(sum(comp_share), 1e-6);
    task_share = node_task_counts / max(sum(node_task_counts), 1);
    conc = max(node_load) + std(node_load) + max(task_share);
end

function mem_pressure = local_memory_pressure_proxy(theta, state)
    node_mem = state.task_mem_mb' * theta;
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    ratio = node_mem ./ max(mem_caps, 1);
    mem_pressure = max(ratio) + mean(max(0, ratio - 0.85)) * 3;
end

function deadline_tail = local_deadline_tail_proxy(theta, state)
    assign = local_assign_vec(theta, state.N);
    deadline_gap = zeros(state.N, 1);
    for i = 1:state.N
        f = assign(i);
        if f == 0
            deadline_gap(i) = 2;
            continue;
        end
        est = state.Pre.Comm(i, f) + state.Pre.Comp(i, f);
        deadline_gap(i) = max(0, est - state.Task(i, 3)) / max(state.Task(i, 3), 1e-3);
    end
    deadline_tail = mean(deadline_gap) + max(deadline_gap);
end

function prop = local_propagation_proxy(theta, state)
    assign = local_assign_vec(theta, state.N);
    prop = 0;
    prop = local_propagation_proxy_from_assign(assign, state);
end

function baseline_prop = local_baseline_propagation_proxy(state)
    assign = local_baseline_assign_for_propagation(state);
    baseline_prop = local_propagation_proxy_from_assign(assign, state);
end

function prop = local_propagation_proxy_from_assign(assign, state)
    if isempty(assign)
        prop = 0;
        return;
    end
    node_counts = accumarray(max(assign, 1), 1, [state.M, 1]);
    node_loads = zeros(state.M, 1);
    node_mem = zeros(state.M, 1);
    for i = 1:state.N
        fi = assign(i);
        if fi <= 0
            continue;
        end
        node_loads(fi) = node_loads(fi) + state.Pre.Comp(i, fi);
        if isfield(state, 'task_mem_mb')
            node_mem(fi) = node_mem(fi) + state.task_mem_mb(i);
        end
    end

    node_mean_load = mean(node_loads(node_loads > 0));
    if ~isfinite(node_mean_load) || node_mean_load <= 0
        node_mean_load = 1;
    end

    prop = 0;
    pair_count = 0;
    for i = 1:state.N
        fi = assign(i);
        if fi <= 0
            prop = prop + 1.5;
            pair_count = pair_count + 1;
            continue;
        end

        load_i = node_loads(fi) / node_mean_load;
        mem_i = 0;
        if isfield(state, 'Fog') && size(state.Fog, 2) >= 8
            mem_cap = max(state.Fog(fi, 8), 1);
            mem_i = node_mem(fi) / mem_cap;
        end

        local_coupling = 0;
        for j = 1:state.N
            if i == j
                continue;
            end
            fj = assign(j);
            if fj <= 0
                continue;
            end
            dist = norm(state.Thing(i, 1:2) - state.Thing(j, 1:2));
            if dist > 70
                continue;
            end
            near_w = 1 / (1 + dist / 20);
            same_node = double(fi == fj);
            comm_gap = abs(state.Pre.Comm(i, fi) - state.Pre.Comm(j, fj));
            comm_w = 1 / (1 + comm_gap / max(state.Pre.Comm(i, fi), 1e-3));
            local_coupling = local_coupling + near_w * ((1.5 * same_node) + 0.35 * comm_w);
            pair_count = pair_count + 1;
        end
        prop = prop + local_coupling * (1 + 0.55 * load_i + 0.35 * mem_i);
    end

    prop = prop / max(pair_count, 1);
end

function assign = local_baseline_assign_for_propagation(state)
    assign = zeros(state.N, 1);
    hotspot_nodes = local_hotspot_nodes(state);
    node_loads = zeros(state.M, 1);
    for i = 1:state.N
        [~, local_rank] = sort(state.Pre.Comm(i, :) + 0.45 * state.Pre.Comp(i, :), 'ascend');
        rank = local_rank;
        if ~isempty(hotspot_nodes)
            rank = unique([hotspot_nodes(:); local_rank(:)], 'stable');
        end
        for k = 1:numel(rank)
            f = rank(k);
            if f < 1 || f > state.M
                continue;
            end
            est_load = node_loads(f) + state.Pre.Comp(i, f);
            if est_load <= 1.45 * mean(state.Pre.Comp(:, f))
                assign(i) = f;
                node_loads(f) = est_load;
                break;
            end
        end
        if assign(i) == 0
            assign(i) = rank(1);
            node_loads(assign(i)) = node_loads(assign(i)) + state.Pre.Comp(i, assign(i));
        end
    end
end

function hotspot_nodes = local_hotspot_nodes(state)
    hotspot_nodes = [];
    if state.M <= 0
        return;
    end
    load_hint = zeros(state.M, 1);
    for f = 1:state.M
        load_hint(f) = mean(state.Pre.Comp(:, f)) + 0.35 * mean(state.Pre.Comm(:, f));
    end
    [~, ord] = sort(load_hint, 'ascend');
    hotspot_nodes = ord(1:min(2, numel(ord)));
end

function assign = local_assign_vec(theta, N)
    assign = zeros(N, 1);
    for i = 1:N
        f = find(theta(i, :) > 0, 1);
        if ~isempty(f)
            assign(i) = f;
        end
    end
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
            return;
        end
        if isfield(breakdown, 'latency_satisfied') && isfinite(breakdown.latency_satisfied)
            value = breakdown.latency_satisfied;
            return;
        end
        if isfield(breakdown, 'latency_admitted_timely') && isfinite(breakdown.latency_admitted_timely)
            value = breakdown.latency_admitted_timely;
        end
    end
end

function value = local_safe_field(s, field_name, default_value)
    if isstruct(s) && isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    else
        value = default_value;
    end
end

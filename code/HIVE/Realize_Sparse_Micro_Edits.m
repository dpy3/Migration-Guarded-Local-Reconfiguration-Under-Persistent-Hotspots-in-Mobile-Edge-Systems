function candidates = Realize_Sparse_Micro_Edits(plan, state, cfg)
% Realize_Sparse_Micro_Edits
% Builds multiple structurally distinct candidate programs per target cell.

    base_theta = local_build_base_theta(state);
    blank_edit = local_blank_edit();
    blank_candidate = local_blank_candidate(blank_edit);

    candidates = repmat(blank_candidate, 0, 1);
    candidates(1) = blank_candidate;
    candidates(1).theta_candidate = base_theta;
    candidates(1).program_mode = 'Base';
    candidates(1).program_variant = 'BaseGreedy';
    candidates(1).program_group = 0;

    if isfield(cfg, 'HIVE_ENABLE_EDITS') && ~cfg.HIVE_ENABLE_EDITS
        return;
    end

    for p = 1:numel(plan.target_cell_ids)
        cell_id = plan.target_cell_ids(p);
        if cell_id < 1 || cell_id > numel(plan.cells)
            continue;
        end

        cell_item = plan.cells(cell_id);
        mode = char(plan.intervention_types{p});
        programs = local_build_program_family(mode, base_theta, cell_item, cell_id, state, cfg, blank_edit, p);
        for k = 1:numel(programs)
            candidates(end + 1) = programs(k); %#ok<AGROW>
        end
    end
end

function programs = local_build_program_family(mode, base_theta, cell_item, cell_id, state, cfg, blank_edit, group_id)
    task_pool = local_rank_cell_tasks(cell_item, state);
    programs = repmat(local_blank_candidate(blank_edit), 0, 1);
    if isempty(task_pool)
        return;
    end
    bottleneck_scores = local_rank_bottleneck_scores(mode, base_theta, cell_item, task_pool, state, cfg);

    switch lower(strtrim(mode))
        case 'drain'
            variants = local_select_outcome_variants_for_bottlenecks(bottleneck_scores);
        case 'split'
            variants = { ...
                struct('name', 'SplitUrgent', 'mix', {{'split'}}, 'intended_outcome_mode', 'latency_guard'), ...
                struct('name', 'SplitWide', 'mix', {{'splitWide'}}, 'intended_outcome_mode', 'latency_dispersion'), ...
                struct('name', 'SplitDrainMix', 'mix', {{'split', 'drain'}}, 'intended_outcome_mode', 'latency_with_load_relief'), ...
                struct('name', 'SplitQuarantineMix', 'mix', {{'split', 'quarantine'}}, 'intended_outcome_mode', 'latency_with_memory_reanchor')};
        case 'swap'
            variants = { ...
                struct('name', 'SwapPair', 'mix', {{'swap'}}, 'intended_outcome_mode', 'paired_rebalance'), ...
                struct('name', 'SwapDrainMix', 'mix', {{'swap', 'drain'}}, 'intended_outcome_mode', 'paired_load_relief'), ...
                struct('name', 'SwapSplitMix', 'mix', {{'swap', 'split'}}, 'intended_outcome_mode', 'paired_latency_relief')};
        case 'quarantine'
            variants = { ...
                struct('name', 'QuarantineHeavy', 'mix', {{'quarantine'}}, 'intended_outcome_mode', 'memory_reanchor'), ...
                struct('name', 'QuarantineWide', 'mix', {{'quarantineWide'}}, 'intended_outcome_mode', 'memory_dispersion'), ...
                struct('name', 'QuarantineDrainMix', 'mix', {{'quarantine', 'drain'}}, 'intended_outcome_mode', 'memory_with_load_relief'), ...
                struct('name', 'QuarantineSwapMix', 'mix', {{'quarantine', 'swap'}}, 'intended_outcome_mode', 'memory_pair_exchange')};
        otherwise
            variants = { ...
                struct('name', 'GenericSplit', 'mix', {{'split'}}, 'intended_outcome_mode', 'generic_latency_relief'), ...
                struct('name', 'GenericDrain', 'mix', {{'drain'}}, 'intended_outcome_mode', 'generic_load_relief'), ...
                struct('name', 'GenericSwap', 'mix', {{'swap'}}, 'intended_outcome_mode', 'generic_pair_rebalance')};
    end

    for v = 1:numel(variants)
        [theta_new, edit_list, edits_done] = local_execute_program(base_theta, task_pool, cell_item, cell_id, state, cfg, variants{v}.mix);
        if edits_done <= 0
            continue;
        end
        candidate = local_blank_candidate(blank_edit);
        candidate.theta_candidate = theta_new;
        candidate.num_edits = edits_done;
        candidate.edit_list = edit_list;
        candidate.edit_cost = edits_done;
        candidate.candidate_label = sprintf('Cell%d_%s_%s', cell_id, char(mode), variants{v}.name);
        candidate.program_mode = char(mode);
        candidate.program_variant = variants{v}.name;
        candidate.program_group = group_id;
        candidate.program_mix = string(variants{v}.mix);
        if isfield(variants{v}, 'bottleneck_type')
            candidate.bottleneck_type = string(variants{v}.bottleneck_type);
        else
            candidate.bottleneck_type = string(local_default_bottleneck_type_for_mode(mode));
        end
        if isfield(variants{v}, 'intended_outcome_mode')
            candidate.intended_outcome_mode = string(variants{v}.intended_outcome_mode);
        else
            candidate.intended_outcome_mode = "baseline";
        end
        programs(end + 1) = candidate; %#ok<AGROW>
    end
    programs = local_filter_novel_candidates(programs, state, cfg);
end

function variants = local_select_outcome_variants_for_bottlenecks(bottleneck_scores)
    variants = {};
    top_k = min(2, numel(bottleneck_scores));
    for k = 1:top_k
        branch = bottleneck_scores(k);
        branch_variants = local_branch_variants(branch.name);
        for v = 1:numel(branch_variants)
            branch_variants{v}.bottleneck_type = branch.name;
            variants{end + 1} = branch_variants{v}; %#ok<AGROW>
        end
    end
end

function variants = local_branch_variants(bottleneck_type)
    switch lower(char(bottleneck_type))
        case 'memory-bound'
            variants = { ...
                struct('name', 'MemoryAggressiveRelief', 'mix', {{'thresholdRescue', 'clusterTriadGuard', 'quarantine'}}, 'intended_outcome_mode', 'aggressive_relief'), ...
                struct('name', 'MemoryConservativeGuard', 'mix', {{'quarantine'}}, 'intended_outcome_mode', 'conservative_guard')};
        case 'latency-bound'
            variants = { ...
                struct('name', 'LatencyAggressiveRelief', 'mix', {{'thresholdRescue', 'clusterTriadGuard', 'split'}}, 'intended_outcome_mode', 'aggressive_relief'), ...
                struct('name', 'LatencyConservativeGuard', 'mix', {{'split'}}, 'intended_outcome_mode', 'conservative_guard')};
        otherwise
            variants = { ...
                struct('name', 'SpilloverAggressiveRelief', 'mix', {{'thresholdRescue', 'clusterTriadGuard', 'twoHopSpillRelief'}}, 'intended_outcome_mode', 'aggressive_relief'), ...
                struct('name', 'SpilloverConservativeGuard', 'mix', {{'clusterDecouple'}}, 'intended_outcome_mode', 'conservative_guard')};
    end
end

function [theta_new, edit_list, edits_done] = local_execute_program(base_theta, task_pool, cell_item, cell_id, state, cfg, op_mix)
    blank_edit = local_blank_edit();
    theta_new = base_theta;
    edit_list = blank_edit([]);
    edits_done = 0;
    program_budget = local_program_budget(cfg, op_mix);
    active_pool = task_pool;

    for op_idx = 1:numel(op_mix)
        op_name = char(op_mix{op_idx});
        remaining_budget = program_budget - edits_done;
        if remaining_budget <= 0
            break;
        end
        [theta_new, new_edits, added] = local_apply_operation(op_name, theta_new, active_pool, cell_item, cell_id, state, cfg, remaining_budget);
        if added <= 0
            continue;
        end
        if isempty(edit_list)
            edit_list = new_edits;
        else
            edit_list = [edit_list, new_edits]; %#ok<AGROW>
        end
        edits_done = edits_done + added;
        active_pool = local_followup_task_pool(task_pool, new_edits, state);
    end

    min_edits = 1;
    if isfield(cfg, 'HIVE_PROGRAM_MIN_EDITS') && ~isempty(cfg.HIVE_PROGRAM_MIN_EDITS)
        min_edits = cfg.HIVE_PROGRAM_MIN_EDITS;
    end
    if strcmpi(local_stress_mode(cfg), 'propagation')
        min_edits = max(min_edits, local_program_min_edits_for_mix(op_mix));
        if local_is_medium_propagation_state(state, cfg)
            min_edits = max(min_edits, local_medium_program_min_edits_for_mix(op_mix));
        end
    end
    if edits_done < min_edits
        theta_new = base_theta;
        edit_list = blank_edit([]);
        edits_done = 0;
    end
end

function next_pool = local_followup_task_pool(task_pool, new_edits, state)
    if isempty(new_edits)
        next_pool = task_pool;
        return;
    end
    edited_tasks = [new_edits.task_id];
    edited_tasks = edited_tasks(edited_tasks > 0);
    if isempty(edited_tasks)
        next_pool = task_pool;
        return;
    end
    next_pool = unique(edited_tasks(:)', 'stable');
    if numel(next_pool) < min(3, numel(task_pool))
        next_pool = local_merge_front(next_pool, task_pool(1:min(4, numel(task_pool))));
    end
    next_pool = local_rank_external_tasks(next_pool, state);
end

function programs = local_filter_novel_candidates(programs, state, cfg)
    if numel(programs) <= 1
        return;
    end
    if ~strcmpi(local_stress_mode(cfg), 'propagation')
        return;
    end

    rows = struct('idx', {}, 'sat', {}, 'lat', {}, 'backlog', {}, 'propdrop', {}, 'edits', {});
    for i = 1:numel(programs)
        rows(i).idx = i; %#ok<AGROW>
        [rows(i).sat, ~, breakdown] = calculate_metrics_v2(programs(i).theta_candidate, state.Task, state.Fog, state.Thing, state.DNN_Data, state.Pre); %#ok<AGROW>
        rows(i).lat = local_pick_latency_from_breakdown(breakdown); %#ok<AGROW>
        rows(i).backlog = local_backlog_proxy_from_theta(programs(i).theta_candidate, state.Pre); %#ok<AGROW>
        rows(i).propdrop = local_preview_propagation_drop(programs(i).theta_candidate, state); %#ok<AGROW>
        rows(i).edits = programs(i).num_edits; %#ok<AGROW>
    end
    stats_tbl = struct2table(rows);
    stats_tbl = sortrows(stats_tbl, {'sat', 'propdrop', 'edits', 'lat'}, {'descend', 'descend', 'ascend', 'ascend'});

    keep = false(height(stats_tbl), 1);
    kept_rows = struct('sat', {}, 'lat', {}, 'backlog', {}, 'propdrop', {}, 'edits', {});
    sat_eps = 0.18;
    prop_eps = 0.045;
    lat_eps = 12.0;
    edit_eps = 2;
    max_keep = 5;

    for r = 1:height(stats_tbl)
        cand = stats_tbl(r, :);
        is_novel = isempty(kept_rows);
        for k = 1:numel(kept_rows)
            if abs(cand.sat - kept_rows(k).sat) >= sat_eps || ...
               abs(cand.propdrop - kept_rows(k).propdrop) >= prop_eps || ...
               abs(cand.lat - kept_rows(k).lat) >= lat_eps || ...
               abs(cand.edits - kept_rows(k).edits) >= edit_eps
                is_novel = true;
                break;
            end
        end
        if is_novel
            keep(r) = true;
            kept_rows(end + 1) = struct( ... %#ok<AGROW>
                'sat', cand.sat, ...
                'lat', cand.lat, ...
                'backlog', cand.backlog, ...
                'propdrop', cand.propdrop, ...
                'edits', cand.edits);
        end
        if nnz(keep) >= max_keep
            break;
        end
    end

    if ~any(keep)
        keep(1) = true;
    end
    programs = programs(stats_tbl.idx(keep));
end

function [theta_new, new_edits, added] = local_apply_operation(op_name, theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    new_edits = blank_edit([]);
    added = 0;
    theta_new = theta_now;

    switch lower(op_name)
        case 'thresholdrescue'
            [theta_new, new_edits, added] = local_apply_threshold_rescue(theta_new, task_pool, cell_item, cell_id, state, cfg, budget);

        case {'drain', 'drainwide'}
            hotspot = local_get_primary_node(theta_new, task_pool, cell_item);
            dst_rank = local_rank_destinations_by_load(hotspot, state, theta_new);
            if strcmpi(op_name, 'drainwide')
                drain_pool = local_merge_front(find(theta_new(:, hotspot) > 0)', task_pool);
            else
                drain_pool = find(theta_new(:, hotspot) > 0)';
            end
            [theta_new, new_edits, added] = local_relocate_tasks(theta_new, drain_pool, hotspot, dst_rank, cell_id, 'Drain', state, cfg, cell_item, 'drain', budget);

        case 'coordinateddrain'
            [theta_new, new_edits, added] = local_apply_coordinated_drain(theta_new, task_pool, cell_item, cell_id, state, cfg, budget);

        case 'twohopspillrelief'
            [theta_new, new_edits, added] = local_apply_twohop_spill_relief(theta_new, task_pool, cell_item, cell_id, state, cfg, budget);

        case 'clusterdecouple'
            [theta_new, new_edits, added] = local_apply_cluster_decouple(theta_new, task_pool, cell_item, cell_id, state, cfg, budget);

        case 'clustertriadguard'
            [theta_new, new_edits, added] = local_apply_cluster_triad_guard(theta_new, task_pool, cell_item, cell_id, state, cfg, budget);

        case {'split', 'splitwide'}
            urgent_pool = task_pool;
            slack = state.deadline_slack(task_pool);
            if ~strcmpi(op_name, 'splitwide')
                urgent_pool = task_pool(slack <= median(slack));
            end
            if isempty(urgent_pool)
                urgent_pool = task_pool;
            end
            [theta_new, new_edits, added] = local_relocate_by_latency(theta_new, urgent_pool, cell_id, 'Split', state, cfg, cell_item, budget);

        case {'quarantine', 'quarantinewide'}
            heavy_pool = task_pool;
            memv = state.task_mem_mb(task_pool);
            if ~strcmpi(op_name, 'quarantinewide')
                heavy_pool = task_pool(memv >= median(memv));
            end
            if isempty(heavy_pool)
                heavy_pool = task_pool;
            end
            dst_rank = local_rank_destinations_by_memory_class(state, theta_new);
            [theta_new, new_edits, added] = local_relocate_tasks(theta_new, heavy_pool, [], dst_rank, cell_id, 'Quarantine', state, cfg, cell_item, 'quarantine', budget);

        case 'swap'
            anchor_node = local_get_primary_node(theta_new, task_pool, cell_item);
            partner_nodes = local_rank_destinations_by_load(anchor_node, state, theta_new);
            [theta_new, new_edits, added] = local_apply_swap_program(theta_new, task_pool, anchor_node, partner_nodes, cell_id, state, cfg, cell_item, budget);
    end
end

function [theta_new, edits, added] = local_apply_threshold_rescue(theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    rescue_pool = local_rank_near_threshold_tasks(task_pool, theta_now, state);
    if isempty(rescue_pool)
        rescue_pool = local_rank_low_slack_tasks(task_pool, state);
    end

    for t = 1:numel(rescue_pool)
        if added >= budget
            break;
        end
        i = rescue_pool(t);
        src = find(theta_new(i, :) > 0, 1);
        if isempty(src)
            continue;
        end
        dst_rank = local_rank_destinations_for_threshold_cross(i, src, theta_new, state);
        dst = local_first_feasible_from_rank(i, src, dst_rank, theta_new, state, cfg, cell_item, 'thresholdrescue');
        if isempty(dst) || dst == src
            continue;
        end
        theta_new(i, src) = 0;
        theta_new(i, dst) = 1;
        added = added + 1;
        edits(added) = local_build_edit_record(i, src, dst, cell_id, 'ThresholdRescue', state, cell_item); %#ok<AGROW>
    end
end

function [theta_new, edits, added] = local_relocate_tasks(theta_now, move_pool, src_lock, dst_rank, cell_id, mode, state, cfg, cell_item, strategy, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    if isempty(move_pool)
        return;
    end
    move_pool = unique(move_pool, 'stable');
    ranked_pool = local_rank_external_tasks(move_pool, state);

    for t = 1:numel(ranked_pool)
        if added >= budget
            break;
        end
        i = ranked_pool(t);
        src = find(theta_new(i, :) > 0, 1);
        if isempty(src)
            continue;
        end
        if ~isempty(src_lock) && src ~= src_lock
            continue;
        end
        dst = local_first_feasible_from_rank(i, src, dst_rank, theta_new, state, cfg, cell_item, strategy);
        if isempty(dst) || dst == src
            continue;
        end
        theta_new(i, src) = 0;
        theta_new(i, dst) = 1;
        added = added + 1;
        edits(added) = local_build_edit_record(i, src, dst, cell_id, mode, state, cell_item); %#ok<AGROW>
    end
end

function [theta_new, edits, added] = local_relocate_by_latency(theta_now, urgent_pool, cell_id, mode, state, cfg, cell_item, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    urgent_pool = local_rank_external_tasks(urgent_pool, state);
    for t = 1:numel(urgent_pool)
        if added >= budget
            break;
        end
        i = urgent_pool(t);
        src = find(theta_new(i, :) > 0, 1);
        if isempty(src)
            continue;
        end
        dst_rank = local_rank_destinations_low_latency(i, src, theta_new, state);
        dst = local_first_feasible_from_rank(i, src, dst_rank, theta_new, state, cfg, cell_item, 'split');
        if isempty(dst) || dst == src
            continue;
        end
        theta_new(i, src) = 0;
        theta_new(i, dst) = 1;
        added = added + 1;
        edits(added) = local_build_edit_record(i, src, dst, cell_id, mode, state, cell_item); %#ok<AGROW>
    end
end

function [theta_new, edits, added] = local_apply_swap_program(theta_now, task_pool, anchor_node, partner_nodes, cell_id, state, cfg, cell_item, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    if budget < 2
        return;
    end
    for pn = 1:numel(partner_nodes)
        partner = partner_nodes(pn);
        [theta_new, edits, added, did_swap] = local_try_swap(theta_new, task_pool, anchor_node, partner, cell_id, state, cfg, cell_item, edits, added, budget);
        if did_swap
            return;
        end
    end
end

function [theta_new, edits, added] = local_apply_coordinated_drain(theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    hotspot = local_get_primary_node(theta_new, task_pool, cell_item);
    neighbor_nodes = local_rank_neighbor_hotspots(task_pool, hotspot, theta_new, state);
    dst_rank = local_rank_destinations_by_load(hotspot, state, theta_new);
    primary_pool = find(theta_new(:, hotspot) > 0)';
    primary_pool = local_merge_front(task_pool, primary_pool);
    primary_budget = max(2, ceil(budget / 2));
    if local_is_medium_propagation_state(state, cfg)
        primary_budget = max(primary_budget, min(budget, 4));
    end
    [theta_new, edits1, add1] = local_relocate_tasks(theta_new, primary_pool, hotspot, dst_rank, cell_id, 'CoordinatedDrain', state, cfg, cell_item, 'drain', primary_budget);
    edits = [edits, edits1]; %#ok<AGROW>
    added = added + add1;

    remain = budget - added;
    for nn = 1:numel(neighbor_nodes)
        if remain <= 0
            break;
        end
        neigh = neighbor_nodes(nn);
        neigh_pool = find(theta_new(:, neigh) > 0)';
        if isempty(neigh_pool)
            continue;
        end
        neigh_dst = local_rank_destinations_by_load(neigh, state, theta_new);
        min_neighbor_budget = 1 + double(numel(neighbor_nodes) > 1);
        take_budget = max(min_neighbor_budget, min(remain, 2 + double(add1 >= 2)));
        [theta_new, edits2, add2] = local_relocate_tasks(theta_new, neigh_pool, neigh, neigh_dst, cell_id, 'CoordinatedDrain', state, cfg, cell_item, 'drain', take_budget);
        edits = [edits, edits2]; %#ok<AGROW>
        added = added + add2;
        remain = budget - added;
    end
end

function [theta_new, edits, added] = local_apply_twohop_spill_relief(theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    hotspot = local_get_primary_node(theta_new, task_pool, cell_item);
    src_tasks = find(theta_new(:, hotspot) > 0)';
    src_tasks = local_merge_front(task_pool, src_tasks);
    src_tasks = local_rank_external_tasks(src_tasks, state);
    first_hop = local_rank_destinations_by_load(hotspot, state, theta_new);

    for t = 1:numel(src_tasks)
        if added >= budget
            break;
        end
        i = src_tasks(t);
        src = find(theta_new(i, :) > 0, 1);
        if isempty(src) || src ~= hotspot
            continue;
        end
        mid = local_first_feasible_from_rank(i, src, first_hop, theta_new, state, cfg, cell_item, 'drain');
        if isempty(mid) || mid == src
            continue;
        end
        second_hop = local_rank_destinations_by_load(mid, state, theta_new);
        dst = local_first_feasible_from_rank(i, mid, second_hop, theta_new, state, cfg, cell_item, 'drain');
        if isempty(dst) || dst == mid || dst == src
            continue;
        end
        min_prop_gain = 0.16;
        if local_is_medium_propagation_state(state, cfg)
            min_prop_gain = 0.12;
        end
        if local_propagation_edit_gain(src, dst, state, cell_item) < min_prop_gain
            continue;
        end
        theta_new(i, src) = 0;
        theta_new(i, dst) = 1;
        added = added + 1;
        edits(added) = local_build_edit_record(i, src, dst, cell_id, 'TwoHopSpillRelief', state, cell_item); %#ok<AGROW>
    end
end

function [theta_new, edits, added] = local_apply_cluster_decouple(theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    hotspot = local_get_primary_node(theta_new, task_pool, cell_item);
    cluster_pool = local_rank_cluster_tasks(task_pool, hotspot, theta_new, state);
    if strcmpi(local_stress_mode(cfg), 'propagation')
        cluster_pool = local_merge_front(cluster_pool, task_pool);
        if local_is_medium_propagation_state(state, cfg)
            hotspot_tasks = find(theta_now(:, hotspot) > 0)';
            cluster_pool = local_merge_front(cluster_pool, hotspot_tasks);
        end
    end
    dst_rank = local_rank_destinations_by_cluster_break(hotspot, cluster_pool, theta_new, state);
    [theta_new, edits, added] = local_relocate_tasks(theta_new, cluster_pool, hotspot, dst_rank, cell_id, 'ClusterDecouple', state, cfg, cell_item, 'drain', budget);
end

function [theta_new, edits, added] = local_apply_cluster_triad_guard(theta_now, task_pool, cell_item, cell_id, state, cfg, budget)
    blank_edit = local_blank_edit();
    edits = blank_edit([]);
    theta_new = theta_now;
    added = 0;
    if budget <= 0
        return;
    end

    hotspot = local_get_primary_node(theta_new, task_pool, cell_item);
    triad_pool = local_rank_cluster_tasks(task_pool, hotspot, theta_new, state);
    triad_pool = local_merge_front(triad_pool, task_pool);
    triad_pool = local_rank_external_tasks(triad_pool, state);

    split_budget = max(2, ceil(0.34 * budget));
    reanchor_budget = max(2, ceil(0.33 * budget));
    guard_budget = max(1, budget - split_budget - reanchor_budget);

    split_rank = local_rank_destinations_by_cluster_split_guard(hotspot, triad_pool, theta_new, state);
    [theta_new, edits1, add1] = local_relocate_tasks(theta_new, triad_pool, hotspot, split_rank, cell_id, 'ClusterTriadGuard', state, cfg, cell_item, 'triadSplit', split_budget);
    edits = [edits, edits1]; %#ok<AGROW>
    added = added + add1;

    remain = budget - added;
    if remain <= 0
        return;
    end
    edited_pool = local_followup_task_pool(triad_pool, edits1, state);
    heavy_pool = local_rank_heavy_edited_tasks(edited_pool, state);
    mem_rank = local_rank_destinations_by_memory_latency_guard(theta_new, state, hotspot, triad_pool);
    take_budget = min(remain, reanchor_budget);
    [theta_new, edits2, add2] = local_relocate_tasks(theta_new, heavy_pool, [], mem_rank, cell_id, 'ClusterTriadGuard', state, cfg, cell_item, 'triadMemory', take_budget);
    edits = [edits, edits2]; %#ok<AGROW>
    added = added + add2;

    remain = budget - added;
    if remain <= 0
        return;
    end
    guard_pool = local_followup_task_pool(local_merge_front(edited_pool, heavy_pool), edits2, state);
    guard_pool = local_rank_low_slack_tasks(guard_pool, state);
    take_budget = min(remain, max(1, guard_budget));
    [theta_new, edits3, add3] = local_relocate_by_latency(theta_new, guard_pool, cell_id, 'ClusterTriadGuard', state, cfg, cell_item, take_budget);
    edits = [edits, edits3]; %#ok<AGROW>
    added = added + add3;
end

function [theta_new, edit_list, edits_done, did_swap] = local_try_swap(theta_new, task_pool, anchor_node, partner_node, cell_id, state, cfg, cell_item, edit_list, edits_done, budget)
    did_swap = false;
    anchor_tasks = task_pool(arrayfun(@(i) any(theta_new(i, anchor_node) > 0), task_pool));
    partner_tasks = find(theta_new(:, partner_node) > 0)';
    if isempty(anchor_tasks) || isempty(partner_tasks)
        return;
    end

    anchor_tasks = local_rank_external_tasks(anchor_tasks, state);
    partner_tasks = local_rank_external_tasks(partner_tasks, state);
    for a = 1:numel(anchor_tasks)
        if edits_done + 2 > budget
            break;
        end
        ta = anchor_tasks(a);
        mem_a = state.task_mem_mb(ta);
        for b = 1:numel(partner_tasks)
            tb = partner_tasks(b);
            mem_b = state.task_mem_mb(tb);
            if ~local_swap_is_feasible(ta, tb, anchor_node, partner_node, theta_new, state, mem_a, mem_b)
                continue;
            end
            before_cost = local_assignment_cost(ta, anchor_node, state) + local_assignment_cost(tb, partner_node, state);
            after_cost = local_assignment_cost(ta, partner_node, state) + local_assignment_cost(tb, anchor_node, state);
            if after_cost > before_cost * 1.25
                continue;
            end
            theta_new(ta, anchor_node) = 0;
            theta_new(ta, partner_node) = 1;
            theta_new(tb, partner_node) = 0;
            theta_new(tb, anchor_node) = 1;
            edits_done = edits_done + 1;
            edit_list(edits_done) = local_build_edit_record(ta, anchor_node, partner_node, cell_id, 'Swap', state, cell_item); %#ok<AGROW>
            edits_done = edits_done + 1;
            edit_list(edits_done) = local_build_edit_record(tb, partner_node, anchor_node, cell_id, 'Swap', state, cell_item); %#ok<AGROW>
            did_swap = true;
            return;
        end
    end
    cfg = cfg; %#ok<NASGU>
end

function base_theta = local_build_base_theta(state)
    M = state.M;
    N = state.N;
    base_theta = zeros(N, M);
    [~, task_order] = sort(state.Task(:, 3), 'ascend');

    node_load = zeros(1, M);
    node_mem = zeros(1, M);
    mem_caps = local_get_node_memory_caps(state.Fog, M);
    for kk = 1:N
        i = task_order(kk);
        [~, rank_idx] = sort(state.Pre.Comm(i, :) + state.Pre.Comp(i, :) + node_load, 'ascend');
        chosen = rank_idx(1);
        for rr = 1:numel(rank_idx)
            f = rank_idx(rr);
            if node_mem(f) + state.task_mem_mb(i) <= mem_caps(f)
                chosen = f;
                break;
            end
        end
        base_theta(i, chosen) = 1;
        node_load(chosen) = node_load(chosen) + state.Pre.Comp(i, chosen);
        node_mem(chosen) = node_mem(chosen) + state.task_mem_mb(i);
    end
end

function task_pool = local_rank_cell_tasks(cell_item, state)
    task_pool = cell_item.task_idx(:)';
    task_pool = local_rank_external_tasks(task_pool, state);
end

function bottleneck_scores = local_rank_bottleneck_scores(mode, theta_now, cell_item, task_pool, state, cfg)
    mode = lower(strtrim(mode));
    if ~strcmpi(mode, 'drain')
        if strcmpi(mode, 'quarantine')
            bottleneck_scores = struct('name', 'memory-bound', 'score', 1);
        elseif strcmpi(mode, 'split')
            bottleneck_scores = struct('name', 'latency-bound', 'score', 1);
        else
            bottleneck_scores = struct('name', 'spillover-bound', 'score', 1);
        end
        return;
    end

    hotspot = local_get_primary_node(theta_now, task_pool, cell_item);
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    node_mem = local_node_memory_usage(theta_now, state.task_mem_mb);
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    hotspot_mem_ratio = node_mem(hotspot) / max(mem_caps(hotspot), 1);
    nonzero_load = node_load(node_load > 0);
    if isempty(nonzero_load)
        load_scale = 1;
    else
        load_scale = mean(nonzero_load);
    end
    hotspot_load_ratio = node_load(hotspot) / max(load_scale, 1e-3);
    cell_slack = state.deadline_slack(task_pool);
    negative_slack = max(0, -cell_slack);
    propagation_score = 0;
    if isfield(cell_item, 'propagation_score') && ~isempty(cell_item.propagation_score)
        propagation_score = cell_item.propagation_score;
    end
    local_fragility = 0;
    if isfield(cell_item, 'local_fragility') && ~isempty(cell_item.local_fragility)
        local_fragility = cell_item.local_fragility;
    end

    memory_score = 1.6 * hotspot_mem_ratio + 0.35 * mean(state.task_mem_mb(task_pool)) / max(mean(state.task_mem_mb), 1e-3);
    latency_score = 1.3 * mean(negative_slack) + 0.55 * max(negative_slack) + 0.20 * hotspot_load_ratio;
    spillover_score = 1.4 * propagation_score + 0.45 * hotspot_load_ratio + 0.20 * max(0, local_fragility);

    if strcmpi(local_stress_mode(cfg), 'memory')
        memory_score = memory_score + 0.35;
    elseif strcmpi(local_stress_mode(cfg), 'deadline')
        latency_score = latency_score + 0.35;
    elseif strcmpi(local_stress_mode(cfg), 'propagation')
        spillover_score = spillover_score + 0.35;
    end

    labels = {'memory-bound', 'latency-bound', 'spillover-bound'};
    values = [memory_score, latency_score, spillover_score];
    [sorted_values, ord] = sort(values, 'descend');
    bottleneck_scores = repmat(struct('name', '', 'score', 0), 1, numel(ord));
    for i = 1:numel(ord)
        bottleneck_scores(i).name = labels{ord(i)};
        bottleneck_scores(i).score = sorted_values(i);
    end
end

function bottleneck_type = local_default_bottleneck_type_for_mode(mode)
    mode = lower(strtrim(mode));
    switch mode
        case 'quarantine'
            bottleneck_type = 'memory-bound';
        case 'split'
            bottleneck_type = 'latency-bound';
        case 'swap'
            bottleneck_type = 'spillover-bound';
        otherwise
            bottleneck_type = 'spillover-bound';
    end
end

function task_pool = local_rank_external_tasks(task_pool, state)
    if isempty(task_pool)
        return;
    end
    task_pool = unique(task_pool(:)', 'stable');
    urgency = -state.deadline_slack(task_pool);
    age = state.task_age(task_pool);
    mem = state.task_mem_mb(task_pool);
    [~, ord] = sortrows([urgency(:), age(:), mem(:)], [-1, -2, -3]);
    task_pool = task_pool(ord);
end

function node_id = local_get_primary_node(theta_now, task_pool, cell_item)
    assigned_nodes = zeros(numel(task_pool), 1);
    for i = 1:numel(task_pool)
        f = find(theta_now(task_pool(i), :) > 0, 1);
        if isempty(f)
            assigned_nodes(i) = cell_item.node_idx(1);
        else
            assigned_nodes(i) = f;
        end
    end
    if isempty(assigned_nodes)
        node_id = cell_item.node_idx(1);
        return;
    end
    node_id = mode(assigned_nodes);
    if isempty(node_id) || node_id == 0
        node_id = assigned_nodes(1);
    end
end

function rank = local_rank_destinations_by_load(src, state, theta_now)
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    [~, ord] = sort(node_load, 'ascend');
    rank = ord(ord ~= src);
end

function rank = local_rank_destinations_by_memory(state, theta_now)
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    mem_use = local_node_memory_usage(theta_now, state.task_mem_mb);
    [~, rank] = sort(mem_use ./ max(mem_caps, 1), 'ascend');
end

function rank = local_rank_destinations_by_memory_class(state, theta_now)
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    mem_use = local_node_memory_usage(theta_now, state.task_mem_mb);
    mem_ratio = mem_use ./ max(mem_caps, 1);
    node_types = ones(1, state.M);
    if isfield(state.Pre, 'Node_Types') && numel(state.Pre.Node_Types) == state.M
        node_types = state.Pre.Node_Types;
    end
    class_score = -1.6 * mem_caps - 0.9 * node_types + 6.0 * mem_ratio;
    [~, rank] = sort(class_score, 'ascend');
end

function rank = local_rank_destinations_low_latency(task_id, src, theta_now, state)
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    score = state.Pre.Comm(task_id, :) + state.Pre.Comp(task_id, :) + 0.20 * node_load;
    [~, ord] = sort(score, 'ascend');
    rank = ord(ord ~= src);
end

function rank = local_rank_neighbor_hotspots(task_pool, hotspot, theta_now, state)
    assign = local_assignment_nodes_for_pool(task_pool, hotspot, theta_now);
    nodes = unique(assign(assign > 0), 'stable')';
    nodes(nodes == hotspot) = [];
    if isempty(nodes)
        rank = [];
        return;
    end
    node_load = sum(state.Pre.Comp(:, nodes), 1);
    [~, ord] = sort(node_load, 'descend');
    rank = nodes(ord);
end

function rank = local_rank_cluster_tasks(task_pool, hotspot, theta_now, state)
    assigned = local_assignment_nodes_for_pool(task_pool, hotspot, theta_now);
    hotspot_pool = task_pool(assigned == hotspot);
    if isempty(hotspot_pool)
        hotspot_pool = task_pool;
    end
    rank = local_rank_external_tasks(hotspot_pool, state);
end

function rank = local_rank_destinations_by_cluster_break(src, cluster_pool, theta_now, state)
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    cluster_center = mean(state.Thing(cluster_pool, 1:2), 1);
    dist_score = zeros(1, state.M);
    for f = 1:state.M
        dist_score(f) = norm(state.Fog(f, 1:2) - cluster_center);
    end
    composite = 0.28 * node_load - 0.06 * dist_score;
    [~, ord] = sort(composite, 'ascend');
    rank = ord(ord ~= src);
end

function rank = local_rank_destinations_by_cluster_split_guard(src, cluster_pool, theta_now, state)
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    mem_use = local_node_memory_usage(theta_now, state.task_mem_mb);
    mem_ratio = mem_use ./ max(mem_caps, 1);
    cluster_center = mean(state.Thing(cluster_pool, 1:2), 1);
    dist_score = zeros(1, state.M);
    for f = 1:state.M
        dist_score(f) = norm(state.Fog(f, 1:2) - cluster_center);
    end
    composite = 0.30 * node_load + 3.0 * mem_ratio - 0.10 * dist_score;
    [~, ord] = sort(composite, 'ascend');
    rank = ord(ord ~= src);
end

function rank = local_rank_destinations_by_memory_latency_guard(theta_now, state, hotspot, cluster_pool)
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    mem_use = local_node_memory_usage(theta_now, state.task_mem_mb);
    mem_ratio = mem_use ./ max(mem_caps, 1);
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    mean_comm = mean(state.Pre.Comm(cluster_pool, :), 1);
    cluster_center = mean(state.Thing(cluster_pool, 1:2), 1);
    geo = zeros(1, state.M);
    for f = 1:state.M
        geo(f) = norm(state.Fog(f, 1:2) - cluster_center);
    end
    hotspot_bias = zeros(1, state.M);
    hotspot_bias(hotspot) = 10;
    composite = 4.0 * mem_ratio + 0.18 * node_load + 0.75 * mean_comm + 0.03 * geo + hotspot_bias;
    [~, rank] = sort(composite, 'ascend');
    rank(rank == hotspot) = [];
end

function rank = local_rank_destinations_for_threshold_cross(task_id, src, theta_now, state)
    node_load = sum(theta_now .* state.Pre.Comp, 1);
    est_total = state.Pre.Comm(task_id, :) + state.Pre.Comp(task_id, :) + 0.18 * node_load;
    deadline = max(state.Task(task_id, 3), 1e-3);
    margin = est_total - deadline;
    same_penalty = zeros(size(margin));
    same_penalty(src) = max(abs(margin)) + 1;
    [~, ord] = sort(margin + same_penalty, 'ascend');
    rank = ord(ord ~= src);
end

function dst = local_first_feasible_from_rank(task_id, src, rank, theta_now, state, cfg, cell_item, strategy)
    dst = src;
    for jj = 1:numel(rank)
        cand = rank(jj);
        if cand == src
            continue;
        end
        if ~local_memory_feasible(task_id, cand, theta_now, state)
            continue;
        end
        if local_edit_is_useful(task_id, src, cand, theta_now, state, cfg, cell_item, strategy)
            dst = cand;
            return;
        end
    end
end

function feasible = local_memory_feasible(task_id, dst, theta_now, state)
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    mem_use = local_node_memory_usage(theta_now, state.task_mem_mb);
    feasible = mem_use(dst) + state.task_mem_mb(task_id) <= mem_caps(dst);
end

function useful = local_edit_is_useful(task_id, src, dst, theta_now, state, cfg, cell_item, strategy)
    src_load = sum(theta_now(:, src) .* state.Pre.Comp(:, src));
    dst_load = sum(theta_now(:, dst) .* state.Pre.Comp(:, dst));
    lat_src = state.Pre.Comm(task_id, src) + state.Pre.Comp(task_id, src) + src_load;
    lat_dst = state.Pre.Comm(task_id, dst) + state.Pre.Comp(task_id, dst) + dst_load;
    frag_gain = cell_item.local_fragility + cell_item.propagation_score;
    slack = max(state.Task(task_id, 3), 1e-3);
    trigger_gain = local_trigger_gain(cfg, strategy);
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    node_mem = local_node_memory_usage(theta_now, state.task_mem_mb);
    src_mem_ratio = node_mem(src) / max(mem_caps(src), 1);
    dst_mem_ratio = node_mem(dst) / max(mem_caps(dst), 1);
    stress_mode = local_stress_mode(cfg);
    max_dest_ratio = 0.88;
    if isfield(cfg, 'HIVE_MEMORY_MAX_DEST_RATIO') && ~isempty(cfg.HIVE_MEMORY_MAX_DEST_RATIO)
        max_dest_ratio = cfg.HIVE_MEMORY_MAX_DEST_RATIO;
    end
    quarantine_lat_slack = 0.22;
    if isfield(cfg, 'HIVE_QUARANTINE_LAT_SLACK') && ~isempty(cfg.HIVE_QUARANTINE_LAT_SLACK)
        quarantine_lat_slack = cfg.HIVE_QUARANTINE_LAT_SLACK;
    end

    switch lower(strategy)
        case 'thresholdrescue'
            useful = (lat_src > 1.01 * slack && lat_dst <= 1.02 * slack) || ...
                (lat_dst + 0.03 * slack < lat_src && lat_dst <= 1.08 * slack) || ...
                (lat_dst < 0.96 * lat_src && dst_mem_ratio <= min(max_dest_ratio, 0.92));
        case 'drain'
            useful = (src_load - dst_load > 0.025 * trigger_gain * max(frag_gain, 0.5)) || ...
                (lat_dst <= lat_src + 0.45 * trigger_gain * slack) || ...
                (strcmpi(stress_mode, 'propagation') && local_propagation_edit_gain(src, dst, state, cell_item) > 0.12 && lat_dst <= lat_src + 0.80 * trigger_gain * slack) || ...
                (strcmpi(stress_mode, 'propagation') && src_load > 1.05 * dst_load) || ...
                (strcmpi(stress_mode, 'propagation') && local_structural_decouple_gain(task_id, src, dst, theta_now, state, cell_item) > local_decouple_threshold(state, cfg));
        case 'split'
            useful = (lat_dst <= min(lat_src, (1.12 - 0.10 * trigger_gain) * slack)) || ...
                (strcmpi(stress_mode, 'deadline') && lat_dst < 1.03 * lat_src);
        case 'quarantine'
            useful = ((state.task_mem_mb(task_id) >= median(state.task_mem_mb)) || strcmpi(stress_mode, 'memory')) && ...
                (dst_mem_ratio <= max_dest_ratio) && ...
                ((dst_mem_ratio + 0.08 < src_mem_ratio) || (lat_dst <= lat_src + quarantine_lat_slack * trigger_gain * slack));
        case 'swap'
            useful = (lat_dst <= lat_src + 0.50 * trigger_gain * slack) || ...
                (src_load - dst_load > 0.02 * trigger_gain * max(frag_gain, 0.5));
        case 'triadsplit'
            useful = (local_structural_decouple_gain(task_id, src, dst, theta_now, state, cell_item) > max(0.14, local_decouple_threshold(state, cfg) - 0.02)) && ...
                (lat_dst <= lat_src + 0.95 * trigger_gain * slack);
        case 'triadmemory'
            useful = (dst_mem_ratio + 0.10 < src_mem_ratio) && ...
                (lat_dst <= lat_src + 0.80 * trigger_gain * slack);
        case 'triadlatency'
            useful = (lat_dst <= min(lat_src, (1.06 + 0.04 * trigger_gain) * slack)) || ...
                ((lat_dst <= 1.02 * lat_src) && local_propagation_edit_gain(src, dst, state, cell_item) > 0.08);
        otherwise
            useful = (lat_dst <= lat_src + 0.30 * trigger_gain * slack) || ...
                (src_load - dst_load > 0.03 * trigger_gain * max(frag_gain, 0.5));
    end
end

function gain = local_propagation_edit_gain(src, dst, state, cell_item)
    gain = 0;
    if isempty(cell_item.task_idx) || size(state.Fog, 2) < 2
        return;
    end
    cluster_center = mean(state.Thing(cell_item.task_idx, 1:2), 1);
    src_dist = norm(state.Fog(src, 1:2) - cluster_center);
    dst_dist = norm(state.Fog(dst, 1:2) - cluster_center);
    gain = (dst_dist - src_dist) / max(src_dist + 10, 1);
end

function gain = local_structural_decouple_gain(task_id, src, dst, theta_now, state, cell_item)
    gain = 0;
    if src == dst || isempty(cell_item.task_idx)
        return;
    end
    cluster_ids = cell_item.task_idx(:);
    same_src = 0;
    same_dst = 0;
    src_load = 0;
    dst_load = 0;
    for k = 1:numel(cluster_ids)
        tid = cluster_ids(k);
        f = find(theta_now(tid, :) > 0, 1);
        if isempty(f)
            continue;
        end
        same_src = same_src + double(f == src);
        same_dst = same_dst + double(f == dst);
    end
    src_load = sum(theta_now(:, src) .* state.Pre.Comp(:, src));
    dst_load = sum(theta_now(:, dst) .* state.Pre.Comp(:, dst));
    scatter_gain = max(0, (same_src - same_dst) / max(numel(cluster_ids), 1));
    load_gain = max(0, (src_load - dst_load) / max(src_load, 1e-3));
    geo_gain = local_propagation_edit_gain(src, dst, state, cell_item);
    mem_relief = 0;
    if isfield(state, 'Fog') && size(state.Fog, 2) >= 8
        mem_caps = local_get_node_memory_caps(state.Fog, state.M);
        node_mem = local_node_memory_usage(theta_now, state.task_mem_mb);
        src_ratio = node_mem(src) / max(mem_caps(src), 1);
        dst_ratio = node_mem(dst) / max(mem_caps(dst), 1);
        mem_relief = max(0, src_ratio - dst_ratio);
    end
    task_bias = 0.05 * double(ismember(task_id, cluster_ids(1:min(end, max(1, ceil(numel(cluster_ids) / 3))))));
    gain = 0.50 * scatter_gain + 0.25 * load_gain + 0.15 * geo_gain + 0.10 * mem_relief + task_bias;
end

function task_pool = local_rank_heavy_edited_tasks(task_pool, state)
    if isempty(task_pool)
        return;
    end
    mem = state.task_mem_mb(task_pool);
    slack = state.deadline_slack(task_pool);
    [~, ord] = sortrows([-mem(:), slack(:)], [1, 2]);
    task_pool = task_pool(ord);
end

function task_pool = local_rank_low_slack_tasks(task_pool, state)
    if isempty(task_pool)
        return;
    end
    slack = state.deadline_slack(task_pool);
    mem = state.task_mem_mb(task_pool);
    [~, ord] = sortrows([slack(:), -mem(:)], [1, 2]);
    task_pool = task_pool(ord);
end

function task_pool = local_rank_near_threshold_tasks(task_pool, theta_now, state)
    if isempty(task_pool)
        return;
    end
    task_pool = unique(task_pool(:)', 'stable');
    gap = inf(size(task_pool));
    rescue_gain = -inf(size(task_pool));
    for k = 1:numel(task_pool)
        i = task_pool(k);
        src = find(theta_now(i, :) > 0, 1);
        if isempty(src)
            continue;
        end
        src_load = sum(theta_now(:, src) .* state.Pre.Comp(:, src));
        lat_src = state.Pre.Comm(i, src) + state.Pre.Comp(i, src) + src_load;
        deadline = max(state.Task(i, 3), 1e-3);
        gap(k) = abs(lat_src - deadline) / deadline;
        dst_rank = local_rank_destinations_for_threshold_cross(i, src, theta_now, state);
        if ~isempty(dst_rank)
            best_dst = dst_rank(1);
            dst_load = sum(theta_now(:, best_dst) .* state.Pre.Comp(:, best_dst));
            lat_dst = state.Pre.Comm(i, best_dst) + state.Pre.Comp(i, best_dst) + dst_load;
            rescue_gain(k) = lat_src - lat_dst;
        end
    end
    keep = gap <= 0.28;
    if any(keep)
        task_pool = task_pool(keep);
        gap = gap(keep);
        rescue_gain = rescue_gain(keep);
    end
    [~, ord] = sortrows([gap(:), -rescue_gain(:)], [1, 2]);
    task_pool = task_pool(ord);
end

function min_edits = local_program_min_edits_for_mix(op_mix)
    mix_text = lower(strjoin(cellstr(string(op_mix)), '-'));
    min_edits = 2;
    if contains(mix_text, 'clustertriadguard')
        min_edits = 5;
    elseif contains(mix_text, 'coordinateddrain')
        min_edits = 4;
    elseif contains(mix_text, 'clusterdecouple')
        min_edits = 3;
    elseif contains(mix_text, 'twohopspillrelief')
        min_edits = 3;
    elseif contains(mix_text, 'drain') && numel(op_mix) > 1
        min_edits = 3;
    end
end

function min_edits = local_medium_program_min_edits_for_mix(op_mix)
    mix_text = lower(strjoin(cellstr(string(op_mix)), '-'));
    min_edits = 2;
    if contains(mix_text, 'clustertriadguard')
        min_edits = 6;
    elseif contains(mix_text, 'clusterdecouple')
        min_edits = 4;
    elseif contains(mix_text, 'twohopspillrelief')
        min_edits = 4;
    elseif contains(mix_text, 'coordinateddrain')
        min_edits = 4;
    elseif contains(mix_text, 'drain')
        min_edits = 3;
    end
end

function threshold = local_decouple_threshold(state, cfg)
    threshold = 0.16;
    if local_is_medium_propagation_state(state, cfg)
        threshold = 0.11;
    end
end

function tf = local_is_medium_propagation_state(state, cfg)
    tf = strcmpi(local_stress_mode(cfg), 'propagation') && isfield(state, 'N') && state.N >= 200 && state.N <= 240;
end

function feasible = local_swap_is_feasible(task_a, task_b, node_a, node_b, theta_now, state, mem_a, mem_b)
    mem_caps = local_get_node_memory_caps(state.Fog, state.M);
    node_mem = local_node_memory_usage(theta_now, state.task_mem_mb);
    mem_a_new = node_mem(node_a) - mem_a + mem_b;
    mem_b_new = node_mem(node_b) - mem_b + mem_a;
    feasible = mem_a_new <= mem_caps(node_a) && mem_b_new <= mem_caps(node_b) && task_a ~= task_b;
end

function assign = local_assignment_nodes_for_pool(task_pool, fallback_node, theta_now)
    assign = zeros(size(task_pool));
    for k = 1:numel(task_pool)
        f = find(theta_now(task_pool(k), :) > 0, 1);
        if isempty(f)
            assign(k) = fallback_node;
        else
            assign(k) = f;
        end
    end
end

function cost = local_assignment_cost(task_id, node_id, state)
    cost = state.Pre.Comm(task_id, node_id) + state.Pre.Comp(task_id, node_id);
end

function backlog = local_backlog_proxy_from_theta(theta, Pre)
    node_base_time = sum(theta .* Pre.Comp, 1);
    backlog = sum(node_base_time .^ 2) / max(numel(node_base_time), 1);
end

function propdrop = local_preview_propagation_drop(theta, state)
    assign = zeros(state.N, 1);
    for i = 1:state.N
        f = find(theta(i, :) > 0, 1);
        if ~isempty(f)
            assign(i) = f;
        end
    end
    prop_now = local_preview_propagation_proxy_from_assign(assign, state);
    base_prop = local_preview_baseline_propagation_proxy(state);
    propdrop = max(0, base_prop - prop_now);
end

function base_prop = local_preview_baseline_propagation_proxy(state)
    assign = zeros(state.N, 1);
    hotspot_nodes = local_preview_hotspot_nodes(state);
    node_loads = zeros(state.M, 1);
    for i = 1:state.N
        [~, local_rank] = sort(state.Pre.Comm(i, :) + 0.45 * state.Pre.Comp(i, :), 'ascend');
        rank = unique([hotspot_nodes(:); local_rank(:)], 'stable');
        for k = 1:numel(rank)
            f = rank(k);
            est_load = node_loads(f) + state.Pre.Comp(i, f);
            if est_load <= 1.5 * mean(state.Pre.Comp(:, f))
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
    base_prop = local_preview_propagation_proxy_from_assign(assign, state);
end

function prop = local_preview_propagation_proxy_from_assign(assign, state)
    if isempty(assign)
        prop = 0;
        return;
    end
    node_loads = zeros(state.M, 1);
    for i = 1:state.N
        fi = assign(i);
        if fi <= 0
            continue;
        end
        node_loads(fi) = node_loads(fi) + state.Pre.Comp(i, fi);
    end
    load_scale = mean(node_loads(node_loads > 0));
    if ~isfinite(load_scale) || load_scale <= 0
        load_scale = 1;
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
        for j = i+1:state.N
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
            load_w = 0.5 * (node_loads(fi) + node_loads(fj)) / load_scale;
            prop = prop + near_w * ((1.55 * same_node) + 0.35 * comm_w) * (1 + 0.45 * load_w);
            pair_count = pair_count + 1;
        end
    end
    prop = prop / max(pair_count, 1);
end

function nodes = local_preview_hotspot_nodes(state)
    score = zeros(state.M, 1);
    for f = 1:state.M
        score(f) = mean(state.Pre.Comp(:, f)) + 0.35 * mean(state.Pre.Comm(:, f));
    end
    [~, ord] = sort(score, 'ascend');
    nodes = ord(1:min(2, numel(ord)));
end

function value = local_pick_latency_from_breakdown(breakdown)
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
    if ~isfinite(value)
        value = 1e4;
    end
end

function edit_record = local_build_edit_record(task_id, src, dst, cell_id, mode, state, cell_item)
    src_mem = state.task_mem_mb(task_id);
    lat_vec = state.Pre.Comm(task_id, :) + state.Pre.Comp(task_id, :);
    edit_record = struct( ...
        'slot_id', 1, ...
        'cell_id', cell_id, ...
        'edit_type', char(mode), ...
        'task_id', task_id, ...
        'src_node', src, ...
        'dst_node', dst, ...
        'estimated_latency_delta', lat_vec(dst) - lat_vec(src), ...
        'estimated_memory_delta', src_mem, ...
        'estimated_fragility_delta', -max(cell_item.local_fragility, 0.1));
end

function node_mem = local_node_memory_usage(theta, task_mem_mb)
    node_mem = task_mem_mb' * theta;
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

function blank_edit = local_blank_edit()
    blank_edit = struct( ...
        'slot_id', 0, ...
        'cell_id', 0, ...
        'edit_type', '', ...
        'task_id', 0, ...
        'src_node', 0, ...
        'dst_node', 0, ...
        'estimated_latency_delta', 0, ...
        'estimated_memory_delta', 0, ...
        'estimated_fragility_delta', 0);
end

function blank_candidate = local_blank_candidate(blank_edit)
    blank_candidate = struct( ...
        'theta_candidate', [], ...
        'num_edits', 0, ...
        'edit_list', blank_edit([]), ...
        'edit_cost', 0, ...
        'candidate_label', 'Base', ...
        'program_mode', 'Base', ...
        'program_variant', 'BaseGreedy', ...
        'program_group', 0, ...
        'program_mix', string("Base"), ...
        'bottleneck_type', string("base"), ...
        'intended_outcome_mode', string("baseline"));
end

function merged = local_merge_front(primary, extra)
    merged = unique([primary(:); extra(:)], 'stable')';
end

function gain = local_trigger_gain(cfg, strategy)
    gain = 1;
    if isfield(cfg, 'HIVE_EDIT_TRIGGER_GAIN') && ~isempty(cfg.HIVE_EDIT_TRIGGER_GAIN)
        gain = cfg.HIVE_EDIT_TRIGGER_GAIN;
    end
    if strcmpi(local_stress_mode(cfg), 'memory') && contains(lower(strategy), 'quarantine')
        gain = gain + 0.35;
    elseif strcmpi(local_stress_mode(cfg), 'deadline') && contains(lower(strategy), 'split')
        gain = gain + 0.35;
    elseif strcmpi(local_stress_mode(cfg), 'propagation') && contains(lower(strategy), 'drain')
        gain = gain + 0.35;
    end
    if isfield(cfg, 'HIVE_AGGRESSIVE_EDIT_BONUS') && ~isempty(cfg.HIVE_AGGRESSIVE_EDIT_BONUS)
        gain = gain + 0.05 * cfg.HIVE_AGGRESSIVE_EDIT_BONUS;
    end
end

function mode = local_stress_mode(cfg)
    mode = 'none';
    if isfield(cfg, 'HIVE_STRESS_MODE') && ~isempty(cfg.HIVE_STRESS_MODE)
        mode = char(cfg.HIVE_STRESS_MODE);
    end
end

function budget = local_program_budget(cfg, op_mix)
    budget = cfg.HIVE_MAX_EDITS;
    mix_text = lower(strjoin(cellstr(string(op_mix)), '-'));
    if contains(mix_text, 'clustertriadguard')
        budget = budget + 3;
    elseif numel(op_mix) > 1
        mix_bonus = 1;
        if isfield(cfg, 'HIVE_PROGRAM_MIX_BONUS') && ~isempty(cfg.HIVE_PROGRAM_MIX_BONUS)
            mix_bonus = cfg.HIVE_PROGRAM_MIX_BONUS;
        end
        budget = budget + mix_bonus;
    end
end

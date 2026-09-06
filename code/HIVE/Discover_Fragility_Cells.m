function cells = Discover_Fragility_Cells(state, cfg)
% Discover_Fragility_Cells
% Heuristic first-pass cell discovery:
% discover task groups around the most stressed nodes and deadline-critical tasks.

    N = state.N;
    M = state.M;
    node_score = cfg.HIVE_QUEUE_W * state.node_queue_proxy(:) + ...
                 cfg.HIVE_LOAD_W * state.node_load(:) + ...
                 cfg.HIVE_MEM_W * max(0, 1 ./ max(state.node_mem_margin(:), 1));
    [~, node_rank] = sort(node_score, 'descend');
    top_nodes = node_rank(1:min(cfg.HIVE_MAX_CELLS, M));

    [~, urgent_rank] = sort(state.deadline_slack(:), 'ascend');
    urgent_pool = urgent_rank(1:min(max(3, ceil(0.2 * N)), N));

    cells = repmat(struct( ...
        'id', 0, ...
        'task_idx', [], ...
        'node_idx', [], ...
        'type', '', ...
        'queue_pressure', 0, ...
        'memory_pressure', 0, ...
        'deadline_pressure', 0, ...
        'load_pressure', 0, ...
        'skew_pressure', 0, ...
        'local_fragility', 0, ...
        'propagation_score', 0, ...
        'neighbor_count', 0, ...
        'priority_score', 0), 0, 1);

    cid = 0;
    for ii = 1:numel(top_nodes)
        f = top_nodes(ii);
        cid = cid + 1;
        [~, near_rank] = sort(state.Pre.Comm(:, f) + state.Pre.Comp(:, f), 'ascend');
        local_pool = unique([near_rank(1:min(5, N)); urgent_pool(:)]);
        if strcmpi(local_stress_mode(cfg), 'propagation')
            near_rank = near_rank(1:min(8, N));
            local_pool = unique([near_rank(:); urgent_pool(:)]);
        end
        q = state.node_queue_proxy(f);
        mem_margin = max(state.node_mem_margin(f), 1);
        mem_p = 1 / mem_margin;
        deadline_p = mean(max(0, -state.deadline_slack(local_pool)));
        load_p = state.node_load(f);
        skew_p = std(state.Pre.Comp(local_pool, :), 0, 2);
        if isempty(skew_p), skew_p = 0; else, skew_p = mean(skew_p); end

        cells(cid).id = cid;
        cells(cid).task_idx = local_pool(:)';
        cells(cid).node_idx = f;
        cells(cid).type = local_type_label(q, mem_p, deadline_p, load_p);
        cells(cid).queue_pressure = q;
        cells(cid).memory_pressure = mem_p;
        cells(cid).deadline_pressure = deadline_p;
        cells(cid).load_pressure = load_p;
        cells(cid).skew_pressure = skew_p;
        cells(cid).local_fragility = ...
            cfg.HIVE_QUEUE_W * q + ...
            cfg.HIVE_MEM_W * mem_p + ...
            cfg.HIVE_DEADLINE_W * deadline_p + ...
            cfg.HIVE_LOAD_W * load_p + ...
            cfg.HIVE_SKEW_W * skew_p;
    end
end

function label = local_type_label(q, mem_p, deadline_p, load_p)
    [~, idx] = max([q, mem_p, deadline_p, load_p]);
    switch idx
        case 1
            label = 'hotspot';
        case 2
            label = 'memory-tight';
        case 3
            label = 'deadline-critical';
        otherwise
            label = 'load-heavy';
    end
end

function mode = local_stress_mode(cfg)
    mode = 'none';
    if isfield(cfg, 'HIVE_STRESS_MODE') && ~isempty(cfg.HIVE_STRESS_MODE)
        mode = char(cfg.HIVE_STRESS_MODE);
    end
end

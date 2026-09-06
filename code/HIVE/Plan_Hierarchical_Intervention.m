function plan = Plan_Hierarchical_Intervention(cells, state, cfg)
% Plan_Hierarchical_Intervention
% Select top-risk cells and assign one intervention type per cell.

    %#ok<INUSD>
    plan = struct();
    plan.cells = cells;
    plan.target_cell_ids = [];
    plan.intervention_types = {};
    plan.candidate_programs = {};

    if isempty(cells)
        return;
    end

    scores = [cells.priority_score];
    [~, ord] = sort(scores, 'descend');
    top_ids = ord(1:min(cfg.HIVE_MAX_TARGET_CELLS, numel(ord)));

    plan.target_cell_ids = top_ids;
    for i = 1:numel(top_ids)
        c = cells(top_ids(i));
        if isfield(cfg, 'HIVE_FORCE_SINGLE_MODE') && ~isempty(cfg.HIVE_FORCE_SINGLE_MODE)
            mode = char(cfg.HIVE_FORCE_SINGLE_MODE);
        elseif strcmpi(local_stress_mode(cfg), 'propagation') && ...
                (c.propagation_score >= 0.60 * max(c.local_fragility, c.memory_pressure) || c.neighbor_count >= 1)
            mode = 'Drain';
        elseif strcmpi(local_stress_mode(cfg), 'memory') && c.memory_pressure >= 0.85 * max([c.queue_pressure, c.deadline_pressure, c.load_pressure])
            mode = 'Quarantine';
        elseif c.deadline_pressure >= max([c.queue_pressure, c.memory_pressure, c.load_pressure])
            mode = 'Split';
        elseif c.memory_pressure >= max([c.queue_pressure, c.deadline_pressure, c.load_pressure])
            mode = 'Quarantine';
        elseif c.queue_pressure >= c.load_pressure
            mode = 'Drain';
        elseif c.skew_pressure > 0.2
            mode = 'Swap';
        else
            mode = 'Buffer';
        end
        plan.intervention_types{i} = mode; %#ok<AGROW>
        plan.candidate_programs{i} = struct('cell_id', top_ids(i), 'mode', mode); %#ok<AGROW>
    end
end

function mode = local_stress_mode(cfg)
    mode = 'none';
    if isfield(cfg, 'HIVE_STRESS_MODE') && ~isempty(cfg.HIVE_STRESS_MODE)
        mode = char(cfg.HIVE_STRESS_MODE);
    end
end

function cells = Estimate_Fragility_Propagation(cells, state, cfg)
% Estimate_Fragility_Propagation
% Assigns pairwise coupling-based propagation scores.

    if isempty(cells)
        return;
    end

    if isfield(cfg, 'HIVE_ENABLE_PROPAGATION') && ~cfg.HIVE_ENABLE_PROPAGATION
        for i = 1:numel(cells)
            cells(i).propagation_score = 0;
            cells(i).neighbor_count = 0;
            cells(i).priority_score = cells(i).local_fragility;
        end
        return;
    end

    for i = 1:numel(cells)
        spill = 0;
        neigh = 0;
        for j = 1:numel(cells)
            if i == j
                continue;
            end
            shared_tasks = numel(intersect(cells(i).task_idx, cells(j).task_idx));
            shared_nodes = numel(intersect(cells(i).node_idx, cells(j).node_idx));
            task_c = shared_tasks / max(1, min(numel(cells(i).task_idx), numel(cells(j).task_idx)));
            node_c = shared_nodes;
            hotspot_c = double(cells(i).node_idx == cells(j).node_idx);

            comm_i_slice = state.Pre.Comm(cells(i).task_idx, cells(i).node_idx);
            comm_j_slice = state.Pre.Comm(cells(j).task_idx, cells(j).node_idx);
            comm_i = mean(comm_i_slice(:));
            comm_j = mean(comm_j_slice(:));
            link_c = 1 / (1 + abs(comm_i - comm_j));
            load_i = mean(state.Pre.Comp(cells(i).task_idx, cells(i).node_idx));
            load_j = mean(state.Pre.Comp(cells(j).task_idx, cells(j).node_idx));
            load_c = 1 / (1 + abs(load_i - load_j));
            locality_c = local_locality_overlap(cells(i).task_idx, cells(j).task_idx, state);

            g = cfg.HIVE_PROP_TASK_W * task_c + ...
                cfg.HIVE_PROP_SHARED_NODE_W * node_c + ...
                cfg.HIVE_PROP_LINK_W * link_c + ...
                cfg.HIVE_PROP_CLUSTER_BONUS * locality_c + ...
                cfg.HIVE_PROP_SHARED_HOTSPOT_BONUS * hotspot_c * load_c;
            if g > 0.05
                spill = spill + g * cells(j).local_fragility;
                neigh = neigh + 1;
            end
        end
        cells(i).propagation_score = spill;
        cells(i).neighbor_count = neigh;
        cells(i).priority_score = cells(i).local_fragility + spill;
    end
end

function ov = local_locality_overlap(task_idx_i, task_idx_j, state)
    loc_i = state.Thing(task_idx_i, 1:2);
    loc_j = state.Thing(task_idx_j, 1:2);
    if isempty(loc_i) || isempty(loc_j)
        ov = 0;
        return;
    end
    ci = mean(loc_i, 1);
    cj = mean(loc_j, 1);
    dist = norm(ci - cj);
    ov = 1 / (1 + dist / 50);
end

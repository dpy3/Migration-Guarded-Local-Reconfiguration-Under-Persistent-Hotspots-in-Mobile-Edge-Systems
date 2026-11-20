function theta_out = DPP_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre_In, cfg)
    [N, ~] = size(Task);
    [M, ~] = size(Fog);

    if nargin < 5 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end

    if nargin < 6 || isempty(cfg)
        cfg = struct();
    end
    cfg = normalize_dpp_cfg(cfg);

    theta_out = zeros(N, M);

    deadlines = Task(:, 3);
    [~, task_order] = sort(deadlines, 'ascend');

    current_node_base_time = zeros(1, M);
    current_node_mem = zeros(1, M);
    tasks_per_node = zeros(1, M);

    mem_caps = get_node_memory_caps_local(Fog, M);

    for k = 1:N
        i = task_order(k);
        task_mem_mb = get_task_mem_mb_local(Task(i, 2), DNN_Data);

        best_node = 1;
        best_score = inf;

        for f = 1:M
            score_f = compute_dpp_score( ...
                i, f, Task, Pre, current_node_base_time, current_node_mem, ...
                tasks_per_node, mem_caps, task_mem_mb, cfg);

            if score_f < best_score
                best_score = score_f;
                best_node = f;
            end
        end

        theta_out(i, best_node) = 1;
        current_node_base_time(best_node) = current_node_base_time(best_node) + Pre.Comp(i, best_node);
        current_node_mem(best_node) = current_node_mem(best_node) + task_mem_mb;
        tasks_per_node(best_node) = tasks_per_node(best_node) + 1;
    end

    row_sum = sum(theta_out, 2);
    assert(all(row_sum == 1), 'DPP theta_out must be one-hot per row.');
end

function cfg = normalize_dpp_cfg(cfg)
    if ~isfield(cfg, 'DPP_V_WEIGHT'), cfg.DPP_V_WEIGHT = 1.0; end
    if ~isfield(cfg, 'DPP_RHO_SOFT'), cfg.DPP_RHO_SOFT = 0.85; end
    if ~isfield(cfg, 'DPP_RHO_HARD'), cfg.DPP_RHO_HARD = 0.99; end
    if ~isfield(cfg, 'DPP_MEM_PENALTY'), cfg.DPP_MEM_PENALTY = 1e4; end
    if ~isfield(cfg, 'DPP_RHO_PENALTY'), cfg.DPP_RHO_PENALTY = 1e4; end
    if ~isfield(cfg, 'DPP_DEADLINE_PENALTY'), cfg.DPP_DEADLINE_PENALTY = 1e3; end
    if ~isfield(cfg, 'DPP_SOFT_RHO_WEIGHT'), cfg.DPP_SOFT_RHO_WEIGHT = 50; end
    if ~isfield(cfg, 'DPP_WAIT_WEIGHT'), cfg.DPP_WAIT_WEIGHT = 1.0; end
    if ~isfield(cfg, 'DPP_CTX_SWITCH_TIME'), cfg.DPP_CTX_SWITCH_TIME = 500e-6; end
end

function score = compute_dpp_score(i, f, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg)
    mean_deadline = mean(Task(:, 3));
    if mean_deadline < 1e-6
        mean_deadline = 1.0;
    end

    comm_time = Pre.Comm(i, f);
    comp_time = Pre.Comp(i, f);

    projected_base_time = current_node_base_time(f) + comp_time;
    projected_rho = projected_base_time / mean_deadline;

    if projected_rho < cfg.DPP_RHO_HARD
        projected_q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        projected_q_factor = 1e4;
    end

    wait_time = comp_time * max(projected_q_factor - 1, 0);
    projected_tasks = tasks_per_node(f) + 1;
    ctx_time = cfg.DPP_CTX_SWITCH_TIME * (projected_tasks ^ 2);

    est_latency = comm_time + comp_time + cfg.DPP_WAIT_WEIGHT * wait_time + ctx_time;
    score = est_latency;

    projected_mem = current_node_mem(f) + task_mem_mb;
    if projected_mem > mem_caps(f)
        score = score + cfg.DPP_MEM_PENALTY;
    end

    if projected_rho >= cfg.DPP_RHO_HARD
        score = score + cfg.DPP_RHO_PENALTY;
    elseif projected_rho >= cfg.DPP_RHO_SOFT
        score = score + cfg.DPP_SOFT_RHO_WEIGHT * (projected_rho - cfg.DPP_RHO_SOFT);
    end

    deadline = Task(i, 3);
    if est_latency > deadline
        score = score + cfg.DPP_DEADLINE_PENALTY * (est_latency - deadline);
    end

    score = score + cfg.DPP_V_WEIGHT * projected_base_time;
end

function mem_mb = get_task_mem_mb_local(t_type, DNN_Data)
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
    else
        mem_mb = 100;
    end
end

function mem_caps = get_node_memory_caps_local(Fog, M)
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

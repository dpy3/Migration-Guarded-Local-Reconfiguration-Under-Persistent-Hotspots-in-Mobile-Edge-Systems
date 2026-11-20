function theta_out = MPC_Rollout_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre_In, cfg)
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
    cfg = normalize_mpc_cfg(cfg, M);

    theta_out = zeros(N, M);

    deadlines = Task(:, 3);
    [~, task_order] = sort(deadlines, 'ascend');

    current_node_base_time = zeros(1, M);
    current_node_mem = zeros(1, M);
    tasks_per_node = zeros(1, M);
    mem_caps = get_node_memory_caps_local(Fog, M);

    for k = 1:N
        task_idx = task_order(k);
        task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);

        immediate_scores = inf(1, M);
        for f = 1:M
            immediate_scores(f) = compute_stage_cost( ...
                task_idx, f, Task, Pre, current_node_base_time, current_node_mem, ...
                tasks_per_node, mem_caps, task_mem_mb, cfg);
        end

        [~, rank_idx] = sort(immediate_scores, 'ascend');
        candidate_nodes = rank_idx(1:min(cfg.MPC_TOPK_NODES, M));

        best_node = candidate_nodes(1);
        best_rollout_cost = inf;

        for c = 1:numel(candidate_nodes)
            f = candidate_nodes(c);
            sim_base_time = current_node_base_time;
            sim_mem = current_node_mem;
            sim_tasks = tasks_per_node;

            sim_base_time(f) = sim_base_time(f) + Pre.Comp(task_idx, f);
            sim_mem(f) = sim_mem(f) + task_mem_mb;
            sim_tasks(f) = sim_tasks(f) + 1;

            rollout_cost = immediate_scores(f) + simulate_future_cost( ...
                k + 1, task_order, Task, DNN_Data, Pre, sim_base_time, sim_mem, ...
                sim_tasks, mem_caps, cfg);

            if rollout_cost < best_rollout_cost
                best_rollout_cost = rollout_cost;
                best_node = f;
            end
        end

        theta_out(task_idx, best_node) = 1;
        current_node_base_time(best_node) = current_node_base_time(best_node) + Pre.Comp(task_idx, best_node);
        current_node_mem(best_node) = current_node_mem(best_node) + task_mem_mb;
        tasks_per_node(best_node) = tasks_per_node(best_node) + 1;
    end

    row_sum = sum(theta_out, 2);
    assert(all(row_sum == 1), 'MPC theta_out must be one-hot per row.');
end

function cfg = normalize_mpc_cfg(cfg, M)
    if ~isfield(cfg, 'MPC_HORIZON'), cfg.MPC_HORIZON = 3; end
    if ~isfield(cfg, 'MPC_TOPK_NODES'), cfg.MPC_TOPK_NODES = 2; end
    if ~isfield(cfg, 'MPC_GAMMA'), cfg.MPC_GAMMA = 0.7; end
    if ~isfield(cfg, 'MPC_RHO_SOFT'), cfg.MPC_RHO_SOFT = 0.85; end
    if ~isfield(cfg, 'MPC_RHO_HARD'), cfg.MPC_RHO_HARD = 0.99; end
    if ~isfield(cfg, 'MPC_MEM_PENALTY'), cfg.MPC_MEM_PENALTY = 1e4; end
    if ~isfield(cfg, 'MPC_RHO_PENALTY'), cfg.MPC_RHO_PENALTY = 2e4; end
    if ~isfield(cfg, 'MPC_DEADLINE_PENALTY'), cfg.MPC_DEADLINE_PENALTY = 2e3; end
    if ~isfield(cfg, 'MPC_SOFT_RHO_WEIGHT'), cfg.MPC_SOFT_RHO_WEIGHT = 100; end
    if ~isfield(cfg, 'MPC_WAIT_WEIGHT'), cfg.MPC_WAIT_WEIGHT = 1.0; end
    if ~isfield(cfg, 'MPC_CTX_SWITCH_TIME'), cfg.MPC_CTX_SWITCH_TIME = 500e-6; end
    if ~isfield(cfg, 'MPC_BACKLOG_WEIGHT'), cfg.MPC_BACKLOG_WEIGHT = 0.8; end
    if ~isfield(cfg, 'MPC_IMBALANCE_WEIGHT'), cfg.MPC_IMBALANCE_WEIGHT = 0.2; end
    cfg.MPC_HORIZON = max(1, round(cfg.MPC_HORIZON));
    cfg.MPC_TOPK_NODES = max(1, min(M, round(cfg.MPC_TOPK_NODES)));
end

function future_cost = simulate_future_cost(start_pos, task_order, Task, DNN_Data, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, cfg)
    future_cost = 0;
    horizon_end = min(numel(task_order), start_pos + cfg.MPC_HORIZON - 2);
    step_idx = 1;

    for pos = start_pos:horizon_end
        task_idx = task_order(pos);
        task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);

        best_step_cost = inf;
        best_node = 1;

        for f = 1:size(Pre.Comp, 2)
            step_cost = compute_stage_cost( ...
                task_idx, f, Task, Pre, current_node_base_time, current_node_mem, ...
                tasks_per_node, mem_caps, task_mem_mb, cfg);

            if step_cost < best_step_cost
                best_step_cost = step_cost;
                best_node = f;
            end
        end

        discount = cfg.MPC_GAMMA ^ step_idx;
        future_cost = future_cost + discount * best_step_cost;

        current_node_base_time(best_node) = current_node_base_time(best_node) + Pre.Comp(task_idx, best_node);
        current_node_mem(best_node) = current_node_mem(best_node) + task_mem_mb;
        tasks_per_node(best_node) = tasks_per_node(best_node) + 1;
        step_idx = step_idx + 1;
    end
end

function score = compute_stage_cost(i, f, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg)
    mean_deadline = mean(Task(:, 3));
    if mean_deadline < 1e-6
        mean_deadline = 1.0;
    end

    comm_time = Pre.Comm(i, f);
    comp_time = Pre.Comp(i, f);

    projected_base_time = current_node_base_time(f) + comp_time;
    projected_rho = projected_base_time / mean_deadline;

    if projected_rho < cfg.MPC_RHO_HARD
        projected_q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        projected_q_factor = 1e4;
    end

    wait_time = comp_time * max(projected_q_factor - 1, 0);
    projected_tasks = tasks_per_node(f) + 1;
    ctx_time = cfg.MPC_CTX_SWITCH_TIME * (projected_tasks ^ 2);
    est_latency = comm_time + comp_time + cfg.MPC_WAIT_WEIGHT * wait_time + ctx_time;

    projected_mem = current_node_mem;
    projected_mem(f) = projected_mem(f) + task_mem_mb;
    projected_base_vector = current_node_base_time;
    projected_base_vector(f) = projected_base_vector(f) + comp_time;

    imbalance_proxy = std(projected_base_vector);
    backlog_proxy = mean(projected_base_vector);
    score = est_latency + cfg.MPC_BACKLOG_WEIGHT * backlog_proxy + cfg.MPC_IMBALANCE_WEIGHT * imbalance_proxy;

    if projected_mem(f) > mem_caps(f)
        score = score + cfg.MPC_MEM_PENALTY;
    end

    if projected_rho >= cfg.MPC_RHO_HARD
        score = score + cfg.MPC_RHO_PENALTY;
    elseif projected_rho >= cfg.MPC_RHO_SOFT
        score = score + cfg.MPC_SOFT_RHO_WEIGHT * (projected_rho - cfg.MPC_RHO_SOFT);
    end

    deadline = Task(i, 3);
    if est_latency > deadline
        score = score + cfg.MPC_DEADLINE_PENALTY * (est_latency - deadline);
    end
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

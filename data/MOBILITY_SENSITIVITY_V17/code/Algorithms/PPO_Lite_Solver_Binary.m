function theta_out = PPO_Lite_Solver_Binary(Task, Fog, Thing, DNN_Data, max_episodes, Pre_In, cfg)
    [N, ~] = size(Task);
    [M, ~] = size(Fog);

    if nargin < 6 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end

    if nargin < 7 || isempty(cfg)
        cfg = struct();
    end
    cfg = normalize_ppo_cfg(cfg, max_episodes, M);

    input_dim = 3 + 3 * M;
    params = init_policy(input_dim, cfg.PPO_HIDDEN_DIM, M);
    adam_state = init_adam(params);

    deadlines = Task(:, 3);
    [~, task_order] = sort(deadlines, 'ascend');
    mem_caps = get_node_memory_caps_local(Fog, M);

    for episode = 1:cfg.PPO_EPISODES
        states = zeros(N, input_dim);
        actions = zeros(N, 1);
        old_log_probs = zeros(N, 1);
        rewards = zeros(N, 1);

        current_node_base_time = zeros(1, M);
        current_node_mem = zeros(1, M);
        tasks_per_node = zeros(1, M);

        for step = 1:N
            task_idx = task_order(step);
            task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
            state = build_state(task_idx, Task, Pre, current_node_base_time, current_node_mem, mem_caps);
            action_mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg);
            [probs, ~] = forward_policy(params, state, action_mask);
            action = sample_action(probs);
            reward = -compute_stage_cost(task_idx, action, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg);

            states(step, :) = state;
            actions(step) = action;
            old_log_probs(step) = log(max(probs(action), 1e-8));
            rewards(step) = reward;

            current_node_base_time(action) = current_node_base_time(action) + Pre.Comp(task_idx, action);
            current_node_mem(action) = current_node_mem(action) + task_mem_mb;
            tasks_per_node(action) = tasks_per_node(action) + 1;
        end

        returns = discount_rewards(rewards, cfg.PPO_GAMMA);
        advantages = returns - mean(returns);
        adv_std = std(advantages);
        if adv_std > 1e-8
            advantages = advantages / adv_std;
        end

        for update_epoch = 1:cfg.PPO_UPDATE_EPOCHS
            grads = zero_like(params);

            for step = 1:N
                rollout_task_idx = task_order(step);
                rollout_task_mem_mb = get_task_mem_mb_local(Task(rollout_task_idx, 2), DNN_Data);
                replay_node_base_time = zeros(1, M);
                replay_node_mem = zeros(1, M);
                replay_tasks_per_node = zeros(1, M);
                for replay_step = 1:(step - 1)
                    prev_task_idx = task_order(replay_step);
                    prev_action = actions(replay_step);
                    prev_task_mem_mb = get_task_mem_mb_local(Task(prev_task_idx, 2), DNN_Data);
                    replay_node_base_time(prev_action) = replay_node_base_time(prev_action) + Pre.Comp(prev_task_idx, prev_action);
                    replay_node_mem(prev_action) = replay_node_mem(prev_action) + prev_task_mem_mb;
                    replay_tasks_per_node(prev_action) = replay_tasks_per_node(prev_action) + 1;
                end
                action_mask = build_action_mask(rollout_task_idx, Task, Pre, replay_node_base_time, replay_node_mem, replay_tasks_per_node, mem_caps, rollout_task_mem_mb, cfg);
                [probs, cache] = forward_policy(params, states(step, :), action_mask);
                action = actions(step);
                new_log_prob = log(max(probs(action), 1e-8));
                ratio = exp(new_log_prob - old_log_probs(step));
                advantage = advantages(step);

                if (advantage >= 0 && ratio > (1 + cfg.PPO_CLIP_EPS)) || (advantage < 0 && ratio < (1 - cfg.PPO_CLIP_EPS))
                    coeff = 0;
                else
                    coeff = ratio * advantage;
                end

                if abs(coeff) > 0
                    grads = accumulate_policy_grad(grads, params, cache, action, coeff);
                end

                if cfg.PPO_ENTROPY_WEIGHT > 0
                    grads = add_entropy_grad(grads, params, cache, cfg.PPO_ENTROPY_WEIGHT);
                end
            end

            grads = scale_grads(grads, 1 / N);
            grads = clip_grads(grads, cfg.PPO_GRAD_CLIP);
            [params, adam_state] = adam_ascent(params, grads, adam_state, cfg.PPO_LEARNING_RATE);
        end
    end

    theta_out = greedy_assign(params, Task, DNN_Data, Pre, task_order, mem_caps, cfg);
    row_sum = sum(theta_out, 2);
    assert(all(row_sum == 1), 'PPO theta_out must be one-hot per row.');
end

function cfg = normalize_ppo_cfg(cfg, max_episodes, M)
    if ~isfield(cfg, 'PPO_EPISODES')
        cfg.PPO_EPISODES = max_episodes;
    end
    if ~isfield(cfg, 'PPO_HIDDEN_DIM'), cfg.PPO_HIDDEN_DIM = 64; end
    if ~isfield(cfg, 'PPO_CLIP_EPS'), cfg.PPO_CLIP_EPS = 0.2; end
    if ~isfield(cfg, 'PPO_LEARNING_RATE'), cfg.PPO_LEARNING_RATE = 1e-3; end
    if ~isfield(cfg, 'PPO_GAMMA'), cfg.PPO_GAMMA = 0.95; end
    if ~isfield(cfg, 'PPO_UPDATE_EPOCHS'), cfg.PPO_UPDATE_EPOCHS = 4; end
    if ~isfield(cfg, 'PPO_ENTROPY_WEIGHT'), cfg.PPO_ENTROPY_WEIGHT = 1e-4; end
    if ~isfield(cfg, 'PPO_GRAD_CLIP'), cfg.PPO_GRAD_CLIP = 5.0; end
    if ~isfield(cfg, 'PPO_RHO_SOFT'), cfg.PPO_RHO_SOFT = 0.85; end
    if ~isfield(cfg, 'PPO_RHO_HARD'), cfg.PPO_RHO_HARD = 0.99; end
    if ~isfield(cfg, 'PPO_MEM_PENALTY'), cfg.PPO_MEM_PENALTY = 1e4; end
    if ~isfield(cfg, 'PPO_RHO_PENALTY'), cfg.PPO_RHO_PENALTY = 2e4; end
    if ~isfield(cfg, 'PPO_DEADLINE_PENALTY'), cfg.PPO_DEADLINE_PENALTY = 2e3; end
    if ~isfield(cfg, 'PPO_SOFT_RHO_WEIGHT'), cfg.PPO_SOFT_RHO_WEIGHT = 100; end
    if ~isfield(cfg, 'PPO_WAIT_WEIGHT'), cfg.PPO_WAIT_WEIGHT = 1.0; end
    if ~isfield(cfg, 'PPO_CTX_SWITCH_TIME'), cfg.PPO_CTX_SWITCH_TIME = 500e-6; end
    if ~isfield(cfg, 'PPO_USE_ACTION_MASK'), cfg.PPO_USE_ACTION_MASK = false; end
    if ~isfield(cfg, 'PPO_MASK_RHO_HARD'), cfg.PPO_MASK_RHO_HARD = cfg.PPO_RHO_HARD; end
    if ~isfield(cfg, 'PPO_MASK_MEM_TOL'), cfg.PPO_MASK_MEM_TOL = 0; end
    if ~isfield(cfg, 'PPO_MASK_DEADLINE_MARGIN'), cfg.PPO_MASK_DEADLINE_MARGIN = 1.05; end
    cfg.PPO_EPISODES = max(1, round(cfg.PPO_EPISODES));
    cfg.PPO_HIDDEN_DIM = max(8, round(cfg.PPO_HIDDEN_DIM));
    if M < 1
        error('PPO requires at least one fog node.');
    end
end

function params = init_policy(input_dim, hidden_dim, output_dim)
    scale1 = sqrt(2 / max(1, input_dim));
    scale2 = sqrt(2 / max(1, hidden_dim));
    params.W1 = randn(hidden_dim, input_dim) * scale1 * 0.1;
    params.b1 = zeros(hidden_dim, 1);
    params.W2 = randn(output_dim, hidden_dim) * scale2 * 0.1;
    params.b2 = zeros(output_dim, 1);
end

function state = build_state(task_idx, Task, Pre, current_node_base_time, current_node_mem, mem_caps)
    data_size_mb = Task(task_idx, 4);
    deadline = Task(task_idx, 3);
    avg_comp = mean(Pre.Comp(task_idx, :));
    mean_deadline = max(mean(Task(:, 3)), 1e-3);

    norm_deadline = deadline / mean_deadline;
    norm_data = data_size_mb / max(1, max(Task(:, 4)));
    norm_comp = avg_comp / max(1e-3, mean(Pre.Comp(:)));
    norm_loads = current_node_base_time / max(mean_deadline, 1e-3);
    norm_mems = current_node_mem ./ max(mem_caps, 1);
    norm_comm = Pre.Comm(task_idx, :) / max(1e-3, mean(Pre.Comm(:)));

    state = [norm_deadline, norm_data, norm_comp, norm_loads, norm_mems, norm_comm];
end

function [probs, cache] = forward_policy(params, state, action_mask)
    if nargin < 3 || isempty(action_mask)
        action_mask = true(size(params.b2));
    end
    state_col = state(:);
    z1 = params.W1 * state_col + params.b1;
    h1 = tanh(z1);
    logits = params.W2 * h1 + params.b2;
    action_mask = logical(action_mask(:));
    if numel(action_mask) ~= numel(logits)
        action_mask = true(size(logits));
    end
    if ~any(action_mask)
        action_mask(:) = true;
    end
    logits(~action_mask) = -1e9;
    logits = logits - max(logits);
    exp_logits = exp(logits);
    probs = exp_logits / sum(exp_logits);

    cache.state = state_col;
    cache.h1 = h1;
    cache.probs = probs;
    cache.mask = action_mask;
end

function action = sample_action(probs)
    cdf = cumsum(probs);
    r = rand();
    action = find(r <= cdf, 1, 'first');
    if isempty(action)
        action = numel(probs);
    end
end

function returns = discount_rewards(rewards, gamma)
    returns = zeros(size(rewards));
    running = 0;
    for idx = numel(rewards):-1:1
        running = rewards(idx) + gamma * running;
        returns(idx) = running;
    end
end

function grads = zero_like(params)
    grads.W1 = zeros(size(params.W1));
    grads.b1 = zeros(size(params.b1));
    grads.W2 = zeros(size(params.W2));
    grads.b2 = zeros(size(params.b2));
end

function grads = scale_grads(grads, factor)
    grads.W1 = grads.W1 * factor;
    grads.b1 = grads.b1 * factor;
    grads.W2 = grads.W2 * factor;
    grads.b2 = grads.b2 * factor;
end

function grads = accumulate_policy_grad(grads, params, cache, action, coeff)
    delta2 = -cache.probs;
    delta2(action) = delta2(action) + 1;
    delta2 = coeff * delta2;

    grads.W2 = grads.W2 + delta2 * cache.h1';
    grads.b2 = grads.b2 + delta2;

    delta1 = (params.W2' * delta2) .* (1 - cache.h1 .^ 2);
    grads.W1 = grads.W1 + delta1 * cache.state';
    grads.b1 = grads.b1 + delta1;
end

function grads = add_entropy_grad(grads, params, cache, weight)
    entropy_target = cache.probs .* (log(max(cache.probs, 1e-8)) + 1);
    mean_target = sum(entropy_target);
    delta2 = weight * cache.probs .* (mean_target - (log(max(cache.probs, 1e-8)) + 1));

    grads.W2 = grads.W2 + delta2 * cache.h1';
    grads.b2 = grads.b2 + delta2;

    delta1 = (params.W2' * delta2) .* (1 - cache.h1 .^ 2);
    grads.W1 = grads.W1 + delta1 * cache.state';
    grads.b1 = grads.b1 + delta1;
end

function grads = clip_grads(grads, clip_value)
    total_norm = sqrt(sum(grads.W1(:).^2) + sum(grads.b1(:).^2) + sum(grads.W2(:).^2) + sum(grads.b2(:).^2));
    if total_norm > clip_value && total_norm > 0
        scale = clip_value / total_norm;
        grads = scale_grads(grads, scale);
    end
end

function state = init_adam(params)
    state.mW1 = zeros(size(params.W1));
    state.vW1 = zeros(size(params.W1));
    state.mb1 = zeros(size(params.b1));
    state.vb1 = zeros(size(params.b1));
    state.mW2 = zeros(size(params.W2));
    state.vW2 = zeros(size(params.W2));
    state.mb2 = zeros(size(params.b2));
    state.vb2 = zeros(size(params.b2));
    state.t = 0;
end

function [params, state] = adam_ascent(params, grads, state, learning_rate)
    beta1 = 0.9;
    beta2 = 0.999;
    eps_val = 1e-8;
    state.t = state.t + 1;

    [params.W1, state.mW1, state.vW1] = adam_update_tensor(params.W1, grads.W1, state.mW1, state.vW1, state.t, learning_rate, beta1, beta2, eps_val);
    [params.b1, state.mb1, state.vb1] = adam_update_tensor(params.b1, grads.b1, state.mb1, state.vb1, state.t, learning_rate, beta1, beta2, eps_val);
    [params.W2, state.mW2, state.vW2] = adam_update_tensor(params.W2, grads.W2, state.mW2, state.vW2, state.t, learning_rate, beta1, beta2, eps_val);
    [params.b2, state.mb2, state.vb2] = adam_update_tensor(params.b2, grads.b2, state.mb2, state.vb2, state.t, learning_rate, beta1, beta2, eps_val);
end

function [param, m, v] = adam_update_tensor(param, grad, m, v, t, learning_rate, beta1, beta2, eps_val)
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * (grad .^ 2);
    m_hat = m / (1 - beta1 ^ t);
    v_hat = v / (1 - beta2 ^ t);
    param = param + learning_rate * m_hat ./ (sqrt(v_hat) + eps_val);
end

function theta_out = greedy_assign(params, Task, DNN_Data, Pre, task_order, mem_caps, cfg)
    N = size(Task, 1);
    M = size(Pre.Comp, 2);
    theta_out = zeros(N, M);
    current_node_base_time = zeros(1, M);
    current_node_mem = zeros(1, M);
    tasks_per_node = zeros(1, M);

    for step = 1:N
        task_idx = task_order(step);
        task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
        state = build_state(task_idx, Task, Pre, current_node_base_time, current_node_mem, mem_caps);
        action_mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg);
        [probs, ~] = forward_policy(params, state, action_mask);

        [~, order] = sort(probs, 'descend');
        best_action = order(1);
        best_cost = inf;

        top_eval = min(numel(order), 3);
        for idx = 1:top_eval
            candidate = order(idx);
            cost = compute_stage_cost(task_idx, candidate, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg);
            if cost < best_cost
                best_cost = cost;
                best_action = candidate;
            end
        end

        theta_out(task_idx, best_action) = 1;
        current_node_base_time(best_action) = current_node_base_time(best_action) + Pre.Comp(task_idx, best_action);
        current_node_mem(best_action) = current_node_mem(best_action) + task_mem_mb;
        tasks_per_node(best_action) = tasks_per_node(best_action) + 1;
    end
end

function action_mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg)
    M = size(Pre.Comp, 2);
    action_mask = true(M, 1);
    if ~cfg.PPO_USE_ACTION_MASK
        return;
    end
    for node_idx = 1:M
        projected_mem = current_node_mem(node_idx) + task_mem_mb;
        projected_base_time = current_node_base_time(node_idx) + Pre.Comp(task_idx, node_idx);
        deadline = Task(task_idx, 3);
        projected_rho = projected_base_time / max(mean(Task(:, 3)), 1e-6);
        projected_tasks = tasks_per_node(node_idx) + 1;
        ctx_time = cfg.PPO_CTX_SWITCH_TIME * (projected_tasks ^ 2);
        if projected_rho < cfg.PPO_RHO_HARD
            projected_q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
        else
            projected_q_factor = 1e4;
        end
        wait_time = Pre.Comp(task_idx, node_idx) * max(projected_q_factor - 1, 0);
        est_latency = Pre.Comm(task_idx, node_idx) + Pre.Comp(task_idx, node_idx) + cfg.PPO_WAIT_WEIGHT * wait_time + ctx_time;
        mem_ok = projected_mem <= (mem_caps(node_idx) + cfg.PPO_MASK_MEM_TOL);
        rho_ok = projected_rho < cfg.PPO_MASK_RHO_HARD;
        ddl_ok = est_latency <= cfg.PPO_MASK_DEADLINE_MARGIN * deadline;
        action_mask(node_idx) = mem_ok && rho_ok && ddl_ok;
    end
    if ~any(action_mask)
        score = inf(M, 1);
        for node_idx = 1:M
            score(node_idx) = compute_stage_cost(task_idx, node_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, task_mem_mb, cfg);
        end
        [~, best_idx] = min(score);
        action_mask(best_idx) = true;
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

    if projected_rho < cfg.PPO_RHO_HARD
        projected_q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        projected_q_factor = 1e4;
    end

    wait_time = comp_time * max(projected_q_factor - 1, 0);
    projected_tasks = tasks_per_node(f) + 1;
    ctx_time = cfg.PPO_CTX_SWITCH_TIME * (projected_tasks ^ 2);
    est_latency = comm_time + comp_time + cfg.PPO_WAIT_WEIGHT * wait_time + ctx_time;
    score = est_latency;

    projected_mem = current_node_mem(f) + task_mem_mb;
    if projected_mem > mem_caps(f)
        score = score + cfg.PPO_MEM_PENALTY;
    end

    if projected_rho >= cfg.PPO_RHO_HARD
        score = score + cfg.PPO_RHO_PENALTY;
    elseif projected_rho >= cfg.PPO_RHO_SOFT
        score = score + cfg.PPO_SOFT_RHO_WEIGHT * (projected_rho - cfg.PPO_RHO_SOFT);
    end

    deadline = Task(i, 3);
    if est_latency > deadline
        score = score + cfg.PPO_DEADLINE_PENALTY * (est_latency - deadline);
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

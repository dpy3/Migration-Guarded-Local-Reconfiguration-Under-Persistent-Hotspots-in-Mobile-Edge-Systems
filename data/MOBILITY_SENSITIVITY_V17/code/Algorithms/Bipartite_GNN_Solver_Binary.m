function [theta_out, params] = Bipartite_GNN_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre_In, cfg)
[M, ~] = size(Fog);
if nargin < 5 || isempty(Pre_In)
    Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
else
    Pre = Pre_In;
end
if nargin < 6 || isempty(cfg)
    cfg = struct();
end
cfg = normalize_gnn_cfg(cfg);
mem_caps = get_node_memory_caps_local(Fog, M);
task_order = get_task_order(Task);
feat_dim = infer_feature_dim(task_order(1), Task, Pre, zeros(1, M), zeros(1, M), zeros(1, M), mem_caps, DNN_Data, cfg);
if isfield(cfg, 'GNN_PRETRAINED_PARAMS') && ~isempty(cfg.GNN_PRETRAINED_PARAMS)
    params = cfg.GNN_PRETRAINED_PARAMS;
    assert(size(params.W1, 1) == feat_dim, 'GNN:FeatureDrift', ...
        'Pretrained feature dimension differs from the current environment.');
else
    params = initialize_params(feat_dim, cfg.GNN_HIDDEN_DIM);
end
if ~cfg.GNN_SKIP_TRAINING
    dataset = collect_imitation_dataset(task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim);
    params = train_graph_policy(params, dataset, cfg);
    params = dagger_finetune_graph_policy(params, dataset, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim);
end
if cfg.GNN_TRAIN_ONLY
    theta_out = [];
    return;
end
theta_out = rollout_graph_policy(params, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, false);
row_sum = sum(theta_out, 2);
assert(all(row_sum == 1), 'GNN theta_out must be one-hot per row.');
end

function cfg = normalize_gnn_cfg(cfg)
if ~isfield(cfg, 'GNN_HIDDEN_DIM'), cfg.GNN_HIDDEN_DIM = 24; end
if ~isfield(cfg, 'GNN_EPOCHS'), cfg.GNN_EPOCHS = 18; end
if ~isfield(cfg, 'GNN_LEARNING_RATE'), cfg.GNN_LEARNING_RATE = 5e-2; end
if ~isfield(cfg, 'GNN_WARMSTART_ROLLOUTS'), cfg.GNN_WARMSTART_ROLLOUTS = 4; end
if ~isfield(cfg, 'GNN_WARMSTART_EPS'), cfg.GNN_WARMSTART_EPS = 0.10; end
if ~isfield(cfg, 'GNN_DAGGER_ITERS'), cfg.GNN_DAGGER_ITERS = 3; end
if ~isfield(cfg, 'GNN_DAGGER_EPOCHS'), cfg.GNN_DAGGER_EPOCHS = 10; end
if ~isfield(cfg, 'GNN_DAGGER_BETA_START'), cfg.GNN_DAGGER_BETA_START = 0.85; end
if ~isfield(cfg, 'GNN_DAGGER_BETA_END'), cfg.GNN_DAGGER_BETA_END = 0.15; end
if ~isfield(cfg, 'GNN_FINETUNE_EPS'), cfg.GNN_FINETUNE_EPS = 0.10; end
if ~isfield(cfg, 'GNN_FINETUNE_WEIGHT'), cfg.GNN_FINETUNE_WEIGHT = 4.0; end
if ~isfield(cfg, 'GNN_DAGGER_AGGREGATE_CAP'), cfg.GNN_DAGGER_AGGREGATE_CAP = 6; end
if ~isfield(cfg, 'GNN_DAGGER_HARDNEG_TOPK'), cfg.GNN_DAGGER_HARDNEG_TOPK = 1; end
if ~isfield(cfg, 'GNN_DAGGER_CONF_WEIGHT'), cfg.GNN_DAGGER_CONF_WEIGHT = 0.25; end
if ~isfield(cfg, 'GNN_DAGGER_MARGIN_WEIGHT'), cfg.GNN_DAGGER_MARGIN_WEIGHT = 0.10; end
if ~isfield(cfg, 'GNN_COST_TARGET_TEMP'), cfg.GNN_COST_TARGET_TEMP = 4.0; end
if ~isfield(cfg, 'GNN_COST_TARGET_MIX'), cfg.GNN_COST_TARGET_MIX = 0.85; end
if ~isfield(cfg, 'GNN_RESIDUAL_SCALE'), cfg.GNN_RESIDUAL_SCALE = 0.15; end
if ~isfield(cfg, 'GNN_RHO_SOFT'), cfg.GNN_RHO_SOFT = 0.85; end
if ~isfield(cfg, 'GNN_RHO_HARD'), cfg.GNN_RHO_HARD = 0.99; end
if ~isfield(cfg, 'GNN_MEM_PENALTY'), cfg.GNN_MEM_PENALTY = 1e4; end
if ~isfield(cfg, 'GNN_RHO_PENALTY'), cfg.GNN_RHO_PENALTY = 1e4; end
if ~isfield(cfg, 'GNN_SOFT_RHO_WEIGHT'), cfg.GNN_SOFT_RHO_WEIGHT = 75; end
if ~isfield(cfg, 'GNN_DEADLINE_PENALTY'), cfg.GNN_DEADLINE_PENALTY = 1e3; end
if ~isfield(cfg, 'GNN_WAIT_WEIGHT'), cfg.GNN_WAIT_WEIGHT = 1.0; end
if ~isfield(cfg, 'GNN_CTX_SWITCH_TIME'), cfg.GNN_CTX_SWITCH_TIME = 500e-6; end
if ~isfield(cfg, 'GNN_MASK_DEADLINE_MARGIN'), cfg.GNN_MASK_DEADLINE_MARGIN = 1.02; end
if ~isfield(cfg, 'GNN_SKIP_TRAINING'), cfg.GNN_SKIP_TRAINING = false; end
if ~isfield(cfg, 'GNN_TRAIN_ONLY'), cfg.GNN_TRAIN_ONLY = false; end
end

function params = initialize_params(feat_dim, hidden_dim)
scale = 0.08;
params.W1 = scale * randn(feat_dim, hidden_dim);
params.b1 = zeros(1, hidden_dim);
params.w2 = scale * randn(hidden_dim, 1);
params.b2 = 0;
params.wr = zeros(feat_dim, 1);
params.br = 0;
end

function order = get_task_order(Task)
[~, order] = sort(Task(:, 3), 'ascend');
end

function feat_dim = infer_feature_dim(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg)
features = build_message_passing_features(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
feat_dim = size(features, 2);
end

function dataset = collect_imitation_dataset(task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim)
N = numel(task_order);
M = size(Fog, 1);
rollouts = max(1, cfg.GNN_WARMSTART_ROLLOUTS);
sample_count = N * rollouts;
X = zeros(sample_count, M, feat_dim);
y = zeros(sample_count, 1);
w = zeros(sample_count, 1);
masks = false(sample_count, M);
targets = zeros(sample_count, M);
cursor = 0;
for rollout_idx = 1:rollouts
    current_node_base_time = zeros(1, M);
    current_node_mem = zeros(1, M);
    tasks_per_node = zeros(1, M);
    eps_now = cfg.GNN_WARMSTART_EPS * (1 - (rollout_idx - 1) / max(1, rollouts));
    for pos = 1:N
        task_idx = task_order(pos);
        mp_features = build_message_passing_features(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
        stage_cost = build_stage_costs(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
        mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
        [~, label_idx] = min(stage_cost);
        cursor = cursor + 1;
        X(cursor, :, :) = mp_features;
        y(cursor) = label_idx;
        w(cursor) = compute_sample_weight(stage_cost, label_idx, cfg, false);
        masks(cursor, :) = mask(:)';
        targets(cursor, :) = build_cost_sensitive_target(stage_cost, mask, label_idx, cfg)';
        action_idx = pick_training_action(stage_cost, label_idx, eps_now);
        [current_node_base_time, current_node_mem, tasks_per_node] = update_state(action_idx, task_idx, current_node_base_time, current_node_mem, tasks_per_node, Pre, Task, DNN_Data);
    end
end
dataset.X = X(1:cursor, :, :);
dataset.y = y(1:cursor);
dataset.w = w(1:cursor);
dataset.mask = masks(1:cursor, :);
dataset.target = targets(1:cursor, :);
end

function action_idx = pick_training_action(stage_cost, label_idx, eps_now)
if rand() >= eps_now
    action_idx = label_idx;
    return;
end
[~, order] = sort(stage_cost, 'ascend');
top_k = min(3, numel(order));
action_idx = order(randi(top_k));
end

function params = train_graph_policy(params, dataset, cfg)
params = train_graph_policy_epochs(params, dataset, cfg, cfg.GNN_EPOCHS);
end

function params = dagger_finetune_graph_policy(params, seed_dataset, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim)
aggregate = seed_dataset;
dagger_iters = max(0, cfg.GNN_DAGGER_ITERS);
for iter = 1:dagger_iters
    beta = schedule_dagger_beta(iter, dagger_iters, cfg);
    rollout_dataset = collect_dagger_dataset(params, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim, beta);
    aggregate = merge_datasets(aggregate, rollout_dataset, cfg);
    params = train_graph_policy_epochs(params, aggregate, cfg, cfg.GNN_DAGGER_EPOCHS);
end
end

function params = train_graph_policy_epochs(params, dataset, cfg, epoch_count)
num_samples = numel(dataset.y);
if num_samples == 0
    return;
end
lr = cfg.GNN_LEARNING_RATE;
for epoch = 1:epoch_count
    order = randperm(num_samples);
    for idx = order
        X = squeeze(dataset.X(idx, :, :));
        label_idx = dataset.y(idx);
        weight = dataset.w(idx);
        mask = [];
        target = [];
        if isfield(dataset, 'mask') && ~isempty(dataset.mask)
            mask = dataset.mask(idx, :)';
        end
        if isfield(dataset, 'target') && ~isempty(dataset.target)
            target = dataset.target(idx, :)';
        end
        [probs, cache] = forward_graph_policy(params, X, mask, cfg);
        if isempty(target)
            target = zeros(size(probs));
            target(label_idx) = 1;
        end
        dlogits = weight * (probs - target);
        grad_w2 = cache.h1' * dlogits;
        grad_b2 = sum(dlogits);
        grad_wr = cache.X' * dlogits;
        grad_br = sum(dlogits);
        dh = (dlogits * params.w2') .* (1 - cache.h1 .^ 2);
        grad_W1 = cache.X' * dh;
        grad_b1 = sum(dh, 1);
        params.W1 = params.W1 - lr * grad_W1;
        params.b1 = params.b1 - lr * grad_b1;
        params.w2 = params.w2 - lr * grad_w2;
        params.b2 = params.b2 - lr * grad_b2;
        params.wr = params.wr - lr * cfg.GNN_RESIDUAL_SCALE * grad_wr;
        params.br = params.br - lr * cfg.GNN_RESIDUAL_SCALE * grad_br;
    end
end
end

function beta = schedule_dagger_beta(iter, total_iters, cfg)
if total_iters <= 1
    beta = cfg.GNN_DAGGER_BETA_END;
    return;
end
t = (iter - 1) / (total_iters - 1);
beta = cfg.GNN_DAGGER_BETA_START + (cfg.GNN_DAGGER_BETA_END - cfg.GNN_DAGGER_BETA_START) * t;
beta = min(max(beta, 0), 1);
end

function dataset = collect_dagger_dataset(params, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, feat_dim, beta)
[N, ~] = size(Task);
M = size(Fog, 1);
X = zeros(N, M, feat_dim);
y = zeros(N, 1);
w = zeros(N, 1);
masks = false(N, M);
targets = zeros(N, M);
current_node_base_time = zeros(1, M);
current_node_mem = zeros(1, M);
tasks_per_node = zeros(1, M);
for pos = 1:N
    task_idx = task_order(pos);
    mp_features = build_message_passing_features(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    stage_cost = build_stage_costs(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    [probs, ~] = forward_graph_policy(params, mp_features, mask, cfg);
    [~, label_idx] = min(stage_cost);
    learner_action = sample_masked_action(probs, mask, stage_cost, cfg.GNN_FINETUNE_EPS);
    expert_action = pick_teacher_action(stage_cost, mask, cfg.GNN_DAGGER_HARDNEG_TOPK);
    teacher_prob = compute_teacher_mixing_prob(probs, mask, beta, cfg);
    if rand() < teacher_prob
        executed_action = expert_action;
    else
        executed_action = learner_action;
    end
    X(pos, :, :) = mp_features;
    y(pos) = label_idx;
    w(pos) = compute_dagger_weight(stage_cost, learner_action, label_idx, cfg);
    masks(pos, :) = mask(:)';
    targets(pos, :) = build_cost_sensitive_target(stage_cost, mask, label_idx, cfg)';
    [current_node_base_time, current_node_mem, tasks_per_node] = update_state(executed_action, task_idx, current_node_base_time, current_node_mem, tasks_per_node, Pre, Task, DNN_Data);
end
dataset = struct('X', X, 'y', y, 'w', w, 'mask', masks, 'target', targets);
end

function dataset_out = merge_datasets(dataset_a, dataset_b, cfg)
if isempty(dataset_a) || ~isfield(dataset_a, 'X') || isempty(dataset_a.X)
    dataset_out = dataset_b;
    return;
end
dataset_out.X = cat(1, dataset_a.X, dataset_b.X);
dataset_out.y = cat(1, dataset_a.y, dataset_b.y);
dataset_out.w = cat(1, dataset_a.w, dataset_b.w);
dataset_out.mask = cat(1, dataset_a.mask, dataset_b.mask);
dataset_out.target = cat(1, dataset_a.target, dataset_b.target);
max_samples = size(dataset_b.X, 1) * max(1, cfg.GNN_DAGGER_AGGREGATE_CAP);
if size(dataset_out.X, 1) > max_samples
    start_idx = size(dataset_out.X, 1) - max_samples + 1;
    dataset_out.X = dataset_out.X(start_idx:end, :, :);
    dataset_out.y = dataset_out.y(start_idx:end);
    dataset_out.w = dataset_out.w(start_idx:end);
    dataset_out.mask = dataset_out.mask(start_idx:end, :);
    dataset_out.target = dataset_out.target(start_idx:end, :);
end
end

function theta_out = rollout_graph_policy(params, task_order, Task, Pre, Fog, DNN_Data, mem_caps, cfg, sample_mode)
if nargin < 9
    sample_mode = false;
end
N = size(Task, 1);
M = size(Fog, 1);
theta_out = zeros(N, M);
current_node_base_time = zeros(1, M);
current_node_mem = zeros(1, M);
tasks_per_node = zeros(1, M);
for pos = 1:numel(task_order)
    task_idx = task_order(pos);
    mp_features = build_message_passing_features(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    stage_cost = build_stage_costs(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
    [probs, ~] = forward_graph_policy(params, mp_features, mask, cfg);
    if sample_mode
        action_idx = sample_masked_action(probs, mask, stage_cost, cfg.GNN_FINETUNE_EPS);
    else
        masked_probs = probs;
        masked_probs(~mask) = -inf;
        if all(~isfinite(masked_probs))
            [~, action_idx] = min(stage_cost);
        else
            [~, action_idx] = max(masked_probs);
        end
    end
    theta_out(task_idx, action_idx) = 1;
    [current_node_base_time, current_node_mem, tasks_per_node] = update_state(action_idx, task_idx, current_node_base_time, current_node_mem, tasks_per_node, Pre, Task, DNN_Data);
end
end

function [probs, cache] = forward_graph_policy(params, X, action_mask, cfg)
if nargin < 3 || isempty(action_mask)
    action_mask = true(size(X, 1), 1);
end
if nargin < 4 || isempty(cfg)
    cfg = struct('GNN_RESIDUAL_SCALE', 0.35);
end
h1 = tanh(X * params.W1 + params.b1);
residual_logits = X * params.wr + params.br;
logits = h1 * params.w2 + params.b2 + cfg.GNN_RESIDUAL_SCALE * residual_logits;
logits = logits(:);
action_mask = logical(action_mask(:));
if ~any(action_mask)
    action_mask(:) = true;
end
logits(~action_mask) = -1e9;
logits = logits - max(logits);
exp_logits = exp(logits);
probs = exp_logits / max(sum(exp_logits), 1e-12);
cache.X = X;
cache.h1 = h1;
cache.mask = action_mask;
cache.residual_logits = residual_logits;
end

function mask = build_action_mask(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg)
M = size(Pre.Comp, 2);
mask = true(M, 1);
task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
mean_deadline = max(mean(Task(:, 3)), 1e-6);
deadline = max(Task(task_idx, 3), 1e-6);
for node_idx = 1:M
    projected_mem = current_node_mem(node_idx) + task_mem_mb;
    projected_base_time = current_node_base_time(node_idx) + Pre.Comp(task_idx, node_idx);
    projected_rho = projected_base_time / mean_deadline;
    projected_tasks = tasks_per_node(node_idx) + 1;
    ctx_time = cfg.GNN_CTX_SWITCH_TIME * (projected_tasks ^ 2);
    if projected_rho < cfg.GNN_RHO_HARD
        q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        q_factor = 1e4;
    end
    wait_time = Pre.Comp(task_idx, node_idx) * max(q_factor - 1, 0);
    est_latency = Pre.Comm(task_idx, node_idx) + Pre.Comp(task_idx, node_idx) + cfg.GNN_WAIT_WEIGHT * wait_time + ctx_time;
    mask(node_idx) = projected_mem <= mem_caps(node_idx) && projected_rho < cfg.GNN_RHO_HARD && est_latency <= cfg.GNN_MASK_DEADLINE_MARGIN * deadline;
end
if ~any(mask)
    mask(:) = true;
end
end

function mp_features = build_message_passing_features(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg)
[task_feat, node_feat, edge_feat] = build_graph_components(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg);
M = size(node_feat, 1);
task_to_node = repmat([task_feat, mean(edge_feat, 1), min(edge_feat, [], 1)], M, 1);
node_round1 = tanh([node_feat, edge_feat, task_to_node, node_feat .* edge_feat(:, 1:size(node_feat, 2))]);
task_round2 = tanh([task_feat, mean(node_round1, 1), max(node_round1, [], 1), std(node_round1, 0, 1)]);
task_round2_rep = repmat(task_round2, M, 1);
node_summary = repmat([mean(node_feat, 1), mean(edge_feat, 1)], M, 1);
mp_features = [node_round1, task_round2_rep, edge_feat, node_summary];
end

function [task_feat, node_feat, edge_feat] = build_graph_components(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg)
M = size(Pre.Comp, 2);
task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
mean_deadline = max(mean(Task(:, 3)), 1e-6);
deadline = max(Task(task_idx, 3), 1e-6);
max_mem_cap = max(mem_caps);
task_type = Task(task_idx, 2);
task_vec = zeros(1, 3);
task_slot = min(max(round(task_type), 1), 3);
task_vec(task_slot) = 1;
remaining_ratio = 1 - (sum(tasks_per_node) / max(1, size(Task, 1)));
task_feat = [task_vec, deadline / mean_deadline, min(Pre.Comp(task_idx, :)) / mean_deadline, mean(Pre.Comp(task_idx, :)) / mean_deadline, min(Pre.Comm(task_idx, :)) / mean_deadline, mean(Pre.Comm(task_idx, :)) / mean_deadline, task_mem_mb / max(max_mem_cap, 1), remaining_ratio];
graph_vec = [mean(current_node_base_time) / mean_deadline, std(current_node_base_time) / mean_deadline, mean(current_node_mem ./ max(mem_caps, 1)), remaining_ratio];
node_feat = zeros(M, 11);
edge_feat = zeros(M, 11);
for node_idx = 1:M
    comp_time = Pre.Comp(task_idx, node_idx);
    comm_time = Pre.Comm(task_idx, node_idx);
    projected_base_time = current_node_base_time(node_idx) + comp_time;
    projected_rho = projected_base_time / mean_deadline;
    projected_mem = current_node_mem(node_idx) + task_mem_mb;
    projected_tasks = tasks_per_node(node_idx) + 1;
    ctx_time = cfg.GNN_CTX_SWITCH_TIME * (projected_tasks ^ 2);
    if projected_rho < cfg.GNN_RHO_HARD
        q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        q_factor = 1e4;
    end
    wait_time = comp_time * max(q_factor - 1, 0);
    est_latency = comm_time + comp_time + cfg.GNN_WAIT_WEIGHT * wait_time + ctx_time;
    slack = (deadline - est_latency) / deadline;
    node_feat(node_idx, :) = [node_idx / max(1, M), mem_caps(node_idx) / max(max_mem_cap, 1), current_node_base_time(node_idx) / mean_deadline, current_node_mem(node_idx) / max(mem_caps(node_idx), 1), tasks_per_node(node_idx) / max(1, size(Task, 1)), mean(Pre.Comp(:, node_idx)) / mean_deadline, mean(Pre.Comm(:, node_idx)) / mean_deadline, graph_vec];
    edge_feat(node_idx, :) = [comp_time / mean_deadline, comm_time / mean_deadline, est_latency / deadline, slack, projected_base_time / mean_deadline, projected_rho, projected_mem / max(mem_caps(node_idx), 1), double(projected_mem > mem_caps(node_idx)), double(projected_rho >= cfg.GNN_RHO_HARD), double(est_latency > deadline), task_mem_mb / max(max_mem_cap, 1)];
end
end

function score = build_stage_costs(task_idx, Task, Pre, current_node_base_time, current_node_mem, tasks_per_node, mem_caps, DNN_Data, cfg)
M = size(Pre.Comp, 2);
score = zeros(M, 1);
task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
mean_deadline = max(mean(Task(:, 3)), 1e-6);
deadline = max(Task(task_idx, 3), 1e-6);
for node_idx = 1:M
    comp_time = Pre.Comp(task_idx, node_idx);
    comm_time = Pre.Comm(task_idx, node_idx);
    projected_base_time = current_node_base_time(node_idx) + comp_time;
    projected_rho = projected_base_time / mean_deadline;
    if projected_rho < cfg.GNN_RHO_HARD
        q_factor = 1 + projected_rho / (2 * max(1e-6, 1 - projected_rho));
    else
        q_factor = 1e4;
    end
    wait_time = comp_time * max(q_factor - 1, 0);
    projected_tasks = tasks_per_node(node_idx) + 1;
    ctx_time = cfg.GNN_CTX_SWITCH_TIME * (projected_tasks ^ 2);
    est_latency = comm_time + comp_time + cfg.GNN_WAIT_WEIGHT * wait_time + ctx_time;
    val = est_latency + projected_base_time;
    projected_mem = current_node_mem(node_idx) + task_mem_mb;
    if projected_mem > mem_caps(node_idx)
        val = val + cfg.GNN_MEM_PENALTY;
    end
    if projected_rho >= cfg.GNN_RHO_HARD
        val = val + cfg.GNN_RHO_PENALTY;
    elseif projected_rho >= cfg.GNN_RHO_SOFT
        val = val + cfg.GNN_SOFT_RHO_WEIGHT * (projected_rho - cfg.GNN_RHO_SOFT);
    end
    if est_latency > deadline
        val = val + cfg.GNN_DEADLINE_PENALTY * (est_latency - deadline);
    end
    score(node_idx) = val;
end
end

function weight = compute_sample_weight(stage_cost, action_idx, cfg, finetune_mode)
best_cost = min(stage_cost);
chosen_cost = stage_cost(action_idx);
gap = max(chosen_cost - best_cost, 0);
scale = max(abs(best_cost), 1);
if finetune_mode
    weight = 1 + cfg.GNN_FINETUNE_WEIGHT * min(gap / scale, 2);
else
    weight = 1 + min(gap / scale, 1);
end
end

function weight = compute_dagger_weight(stage_cost, learner_action, label_idx, cfg)
best_cost = min(stage_cost);
learner_cost = stage_cost(learner_action);
teacher_cost = stage_cost(label_idx);
regret_gap = max(learner_cost - teacher_cost, 0);
scale = max(abs(best_cost), 1);
weight = 1 + cfg.GNN_FINETUNE_WEIGHT * min(regret_gap / scale, 4);
end

function teacher_prob = compute_teacher_mixing_prob(probs, mask, beta, cfg)
masked_probs = probs(:);
masked_probs(~mask(:)) = 0;
prob_sum = sum(masked_probs);
if prob_sum <= 1e-12
    teacher_prob = 1;
    return;
end
masked_probs = masked_probs / prob_sum;
sorted_probs = sort(masked_probs(mask(:)), 'descend');
top1 = sorted_probs(1);
if numel(sorted_probs) >= 2
    margin = top1 - sorted_probs(2);
else
    margin = top1;
end
confidence = max(0, min(1, top1));
margin_conf = max(0, min(1, margin));
teacher_prob = beta + cfg.GNN_DAGGER_CONF_WEIGHT * (1 - confidence) + cfg.GNN_DAGGER_MARGIN_WEIGHT * (1 - margin_conf);
teacher_prob = min(max(teacher_prob, 0), 1);
end

function target = build_cost_sensitive_target(stage_cost, mask, label_idx, cfg)
masked_cost = stage_cost(:);
masked_cost(~mask(:)) = inf;
finite_idx = find(isfinite(masked_cost));
target = zeros(size(masked_cost));
if isempty(finite_idx)
    target(label_idx) = 1;
    return;
end
finite_cost = masked_cost(finite_idx);
cost_min = min(finite_cost);
cost_scale = max(std(finite_cost), max(abs(cost_min), 1));
scaled = (finite_cost - cost_min) / max(cost_scale, 1e-6);
cost_probs = exp(-cfg.GNN_COST_TARGET_TEMP * scaled);
cost_probs = cost_probs / max(sum(cost_probs), 1e-12);
target(finite_idx) = (1 - cfg.GNN_COST_TARGET_MIX) * cost_probs;
target(label_idx) = target(label_idx) + cfg.GNN_COST_TARGET_MIX;
target = target / max(sum(target), 1e-12);
end

function action_idx = pick_teacher_action(stage_cost, mask, top_k)
masked_cost = stage_cost(:);
masked_cost(~mask(:)) = inf;
[~, order] = sort(masked_cost, 'ascend');
finite_order = order(isfinite(masked_cost(order)));
if isempty(finite_order)
    [~, action_idx] = min(stage_cost);
    return;
end
top_k = min(max(1, top_k), numel(finite_order));
action_idx = finite_order(randi(top_k));
end

function action_idx = sample_masked_action(probs, mask, stage_cost, eps_value)
masked_probs = probs(:);
masked_probs(~mask(:)) = 0;
prob_sum = sum(masked_probs);
if prob_sum <= 1e-12
    [~, action_idx] = min(stage_cost);
    return;
end
masked_probs = masked_probs / prob_sum;
if rand() < eps_value
    valid_idx = find(mask(:));
    action_idx = valid_idx(randi(numel(valid_idx)));
    return;
end
cdf = cumsum(masked_probs);
r = rand();
action_idx = find(cdf >= r, 1, 'first');
if isempty(action_idx)
    [~, action_idx] = max(masked_probs);
end
end

function [current_node_base_time, current_node_mem, tasks_per_node] = update_state(action_idx, task_idx, current_node_base_time, current_node_mem, tasks_per_node, Pre, Task, DNN_Data)
task_mem_mb = get_task_mem_mb_local(Task(task_idx, 2), DNN_Data);
current_node_base_time(action_idx) = current_node_base_time(action_idx) + Pre.Comp(task_idx, action_idx);
current_node_mem(action_idx) = current_node_mem(action_idx) + task_mem_mb;
tasks_per_node(action_idx) = tasks_per_node(action_idx) + 1;
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
mem_caps = reshape(mem_caps, 1, []);
if numel(mem_caps) < M
    mem_caps = [mem_caps, inf(1, M - numel(mem_caps))];
else
    mem_caps = mem_caps(1:M);
end
end

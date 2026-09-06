function [avg_satis, viol_rate, breakdown, Latency_Vec, Sat_Vec, Viol_Vec] = calculate_metrics_v2(theta, Task, Fog, ~, DNN_Data, Pre)
    [N, M] = size(theta);
    if nargin < 6
        error('calculate_metrics_v2 requires Pre as the 6th input.');
    end

    violation_mode = 'composite';
    if isfield(Pre, 'Violation_Mode') && ~isempty(Pre.Violation_Mode)
        violation_mode = lower(char(string(Pre.Violation_Mode)));
    end

    theta_bin = zeros(N, M);
    assign_node = zeros(N, 1);
    positive_mask = theta > 0;
    row_positive_cnt = sum(positive_mask, 2);
    invalid_assign_mask = (row_positive_cnt ~= 1);

    for i = 1:N
        if row_positive_cnt(i) > 0
            [~, f] = max(theta(i, :));
            theta_bin(i, f) = 1;
            assign_node(i) = f;
        end
    end

    mean_deadline = mean(Task(:, 3));
    if mean_deadline < 0.1
        mean_deadline = 1.0;
    end

    node_base_time = sum(theta_bin .* Pre.Comp, 1);
    rho = node_base_time / mean_deadline;

    q_factor = ones(1, M);
    stable_mask = rho < 0.99;
    rho_stable = rho(stable_mask);
    q_factor(stable_mask) = 1 + rho_stable ./ (2 * (1 - rho_stable + 1e-6));
    q_factor(~stable_mask) = 1e4;

    tasks_per_node = sum(theta_bin, 1);
    ctx_overhead = 500e-6 * (tasks_per_node .^ 2);

    task_mem_mb = zeros(N, 1);
    for i = 1:N
        task_mem_mb(i) = get_task_mem_mb(Task(i, 2), DNN_Data);
    end

    node_mem = zeros(1, M);
    for i = 1:N
        f = assign_node(i);
        if f > 0
            node_mem(f) = node_mem(f) + task_mem_mb(i);
        end
    end

    mem_caps = get_node_memory_caps(Fog, M);
    mem_viol_mask = node_mem > mem_caps;
    rho_viol_mask = rho >= 0.99;
    node_viol_mask = mem_viol_mask | rho_viol_mask;

    Latency_Vec = inf(N, 1);
    Sat_Vec = zeros(N, 1);
    Admitted_Vec = zeros(N, 1);
    Timely_Vec = zeros(N, 1);
    Resource_Viol_Vec = zeros(N, 1);
    Deadline_Viol_Vec = zeros(N, 1);
    Composite_Viol_Vec = zeros(N, 1);
    Comm_Vec = zeros(N, 1);
    Comp_Vec = zeros(N, 1);
    Wait_Vec = zeros(N, 1);
    Ctx_Vec = zeros(N, 1);

    for i = 1:N
        f = assign_node(i);
        admitted_i = (f > 0) && ~invalid_assign_mask(i);

        if ~admitted_i
            Deadline_Viol_Vec(i) = 1;
            Composite_Viol_Vec(i) = 1;
            continue;
        end

        Admitted_Vec(i) = 1;

        if node_viol_mask(f)
            Resource_Viol_Vec(i) = 1;
        end

        Comm_Vec(i) = Pre.Comm(i, f);
        Comp_Vec(i) = Pre.Comp(i, f);
        Wait_Vec(i) = Pre.Comp(i, f) * max(q_factor(f) - 1, 0);
        Ctx_Vec(i) = ctx_overhead(f);
        Latency_Vec(i) = Comm_Vec(i) + Comp_Vec(i) + Wait_Vec(i) + Ctx_Vec(i);

        if Latency_Vec(i) <= Task(i, 3)
            Timely_Vec(i) = 1;
        else
            Deadline_Viol_Vec(i) = 1;
        end

        Composite_Viol_Vec(i) = max(Resource_Viol_Vec(i), Deadline_Viol_Vec(i));

        if Timely_Vec(i) == 1 && Resource_Viol_Vec(i) == 0
            Sat_Vec(i) = 1;
        end
    end

    switch violation_mode
        case 'deadline'
            Viol_Vec = Deadline_Viol_Vec;
        case 'resource'
            Viol_Vec = Resource_Viol_Vec;
        otherwise
            Viol_Vec = Composite_Viol_Vec;
            violation_mode = 'composite';
    end

    avg_satis = mean(Sat_Vec) * 100;
    viol_rate = mean(Viol_Vec) * 100;

    admitted_mask = Admitted_Vec > 0;
    timely_mask = Timely_Vec > 0;
    satisfied_mask = Sat_Vec > 0;
    timely_admitted_mask = admitted_mask & timely_mask;
    [Memory_Block_Vec, Intrinsic_Latency_Block_Vec, Spillover_Block_Vec, ...
        Feasible_Missed_Vec] = classify_unsatisfied_causes(Sat_Vec, assign_node, ...
        task_mem_mb, node_mem, mem_caps, rho, tasks_per_node, Pre, Task, mean_deadline);

    breakdown.satisfaction_rate = avg_satis;
    breakdown.admission_rate = mean(Admitted_Vec) * 100;
    if any(admitted_mask)
        breakdown.timely_rate_among_admitted = mean(Timely_Vec(admitted_mask)) * 100;
    else
        breakdown.timely_rate_among_admitted = 0;
    end
    breakdown.resource_viol_rate = mean(Resource_Viol_Vec) * 100;
    breakdown.deadline_viol_rate = mean(Deadline_Viol_Vec) * 100;
    breakdown.composite_viol_rate = mean(Composite_Viol_Vec) * 100;
    breakdown.invalid_assign_rate = mean(invalid_assign_mask) * 100;
    breakdown.violation_mode = violation_mode;
    breakdown.latency_admitted_all = safe_mean(Latency_Vec(admitted_mask));
    breakdown.latency_admitted_timely = safe_mean(Latency_Vec(timely_admitted_mask));
    breakdown.latency_satisfied = safe_mean(Latency_Vec(satisfied_mask));
    breakdown.comm = safe_mean(Comm_Vec(admitted_mask));
    breakdown.comp = safe_mean(Comp_Vec(admitted_mask));
    breakdown.wait = safe_mean(Wait_Vec(admitted_mask));
    breakdown.ctx = safe_mean(Ctx_Vec(admitted_mask));
    breakdown.memory_block_count = sum(Memory_Block_Vec);
    breakdown.intrinsic_latency_block_count = sum(Intrinsic_Latency_Block_Vec);
    breakdown.spillover_block_count = sum(Spillover_Block_Vec);
    breakdown.feasible_missed_count = sum(Feasible_Missed_Vec);
    breakdown.admitted_deadline_miss_count = sum(admitted_mask & ~timely_mask);
    breakdown.tasks_per_node = tasks_per_node;
    breakdown.node_mem = node_mem;
    breakdown.node_mem_caps = mem_caps;
    breakdown.rho = rho;
    breakdown.q_factor = q_factor;
    breakdown.detail = struct( ...
        'assign_node', assign_node, ...
        'admitted_vec', Admitted_Vec, ...
        'timely_vec', Timely_Vec, ...
        'satisfied_vec', Sat_Vec, ...
        'resource_viol_vec', Resource_Viol_Vec, ...
        'deadline_viol_vec', Deadline_Viol_Vec, ...
        'composite_viol_vec', Composite_Viol_Vec, ...
        'invalid_assign_mask', invalid_assign_mask, ...
        'comm_vec', Comm_Vec, ...
        'comp_vec', Comp_Vec, ...
        'wait_vec', Wait_Vec, ...
        'ctx_vec', Ctx_Vec);
    breakdown.detail.memory_block_vec = Memory_Block_Vec;
    breakdown.detail.intrinsic_latency_block_vec = Intrinsic_Latency_Block_Vec;
    breakdown.detail.spillover_block_vec = Spillover_Block_Vec;
    breakdown.detail.feasible_missed_vec = Feasible_Missed_Vec;
end

function [Memory_Block_Vec, Intrinsic_Latency_Block_Vec, Spillover_Block_Vec, ...
    Feasible_Missed_Vec] = classify_unsatisfied_causes(Sat_Vec, assign_node, ...
    task_mem_mb, node_mem, mem_caps, rho, tasks_per_node, Pre, Task, mean_deadline)
    N = numel(Sat_Vec);
    M = numel(mem_caps);
    Memory_Block_Vec = zeros(N, 1);
    Intrinsic_Latency_Block_Vec = zeros(N, 1);
    Spillover_Block_Vec = zeros(N, 1);
    Feasible_Missed_Vec = zeros(N, 1);
    node_base_time = rho * mean_deadline;

    for i = 1:N
        if Sat_Vec(i) > 0
            continue;
        end

        mem_without = node_mem;
        base_without = node_base_time;
        count_without = tasks_per_node;
        current_f = assign_node(i);
        if current_f > 0
            mem_without(current_f) = max(0, mem_without(current_f) - task_mem_mb(i));
            base_without(current_f) = max(0, base_without(current_f) - Pre.Comp(i, current_f));
            count_without(current_f) = max(0, count_without(current_f) - 1);
        end

        memory_feasible = (mem_without + task_mem_mb(i)) <= mem_caps;
        if ~any(memory_feasible)
            Memory_Block_Vec(i) = 1;
            continue;
        end

        intrinsic_latency = Pre.Comm(i, :) + Pre.Comp(i, :);
        intrinsic_feasible = memory_feasible & intrinsic_latency <= Task(i, 3);
        if ~any(intrinsic_feasible)
            Intrinsic_Latency_Block_Vec(i) = 1;
            continue;
        end

        loaded_feasible = false;
        candidates = find(intrinsic_feasible);
        for f = candidates
            rho_new = (base_without(f) + Pre.Comp(i, f)) / mean_deadline;
            if rho_new >= 0.99
                continue;
            end
            q_new = 1 + rho_new / (2 * (1 - rho_new + 1e-6));
            ctx_new = 500e-6 * (count_without(f) + 1)^2;
            loaded_latency = Pre.Comm(i, f) + Pre.Comp(i, f) * q_new + ctx_new;
            if loaded_latency <= Task(i, 3)
                loaded_feasible = true;
                break;
            end
        end

        if loaded_feasible
            Feasible_Missed_Vec(i) = 1;
        else
            Spillover_Block_Vec(i) = 1;
        end
    end
end

function mem_mb = get_task_mem_mb(t_type, DNN_Data)
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

    error('DNN_Data must be numeric, cell, or struct array');
end

function value = safe_mean(vec)
    if isempty(vec)
        value = NaN;
        return;
    end
    finite_vec = vec(isfinite(vec));
    if isempty(finite_vec)
        value = NaN;
    else
        value = mean(finite_vec);
    end
end

function mem_caps = get_node_memory_caps(Fog, M)
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

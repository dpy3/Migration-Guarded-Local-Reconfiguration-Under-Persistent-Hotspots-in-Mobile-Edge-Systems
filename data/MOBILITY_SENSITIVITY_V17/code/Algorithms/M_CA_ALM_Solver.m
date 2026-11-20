function [theta_bin, loss_hist, theta_logits] = M_CA_ALM_Solver(Task, Fog, Thing, DNN_Data, max_iter, p_init, theta_init, beta_util, use_repair, Pre_In)
    % M_CA_ALM_Solver.m (Refactored for Hybrid Memetic Algorithm)
    % Memetic Congestion-Aware Augmented Lagrangian Method
    % Features:
    % 1. Integer-Based Evolutionary Framework (Robust Exploration)
    % 2. Gradient-Based Local Search (Lamarckian Learning via Logits)
    % 3. Adaptive Initialization (Greedy/Random)
    % 4. G3R Repair + Global Optimization
    
    if nargin < 9, use_repair = true; end 
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    if nargin >= 10 && ~isempty(Pre_In)
        Pre = Pre_In;
    else
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    end

    % --- Parameters ---
    if nargin >= 8 && ~isempty(beta_util) && isstruct(beta_util)
         params = beta_util;
    else
         params.rho = 0.01; 
         params.mu = 1.02;  
         params.lr = 0.05;
         params.alpha = 0.5;
    end
    
    % Memetic Parameters
    POP_SIZE = 50; % Standard size
    if isfield(params, 'pop_size'), POP_SIZE = params.pop_size; end
    
    MAX_GEN = 50;
    if isfield(params, 'max_gen'), MAX_GEN = params.max_gen; end

    greedy_init_ratio = 0.2;
    if isfield(params, 'greedy_init_ratio'), greedy_init_ratio = params.greedy_init_ratio; end

    greedy_perturb_ratio = 0.1;
    if isfield(params, 'greedy_perturb_ratio'), greedy_perturb_ratio = params.greedy_perturb_ratio; end

    local_search_prob = 0.5;
    if isfield(params, 'local_search_prob'), local_search_prob = params.local_search_prob; end

    guarded_local_search = false;
    if isfield(params, 'guarded_local_search'), guarded_local_search = logical(params.guarded_local_search); end

    repair_aware_fitness = false;
    if isfield(params, 'repair_aware_fitness'), repair_aware_fitness = logical(params.repair_aware_fitness); end

    enable_internal_global_opt = use_repair;
    if isfield(params, 'enable_internal_global_opt'), enable_internal_global_opt = params.enable_internal_global_opt; end
    
    % ALM Steps per Generation
    ALM_STEPS = max(5, round(max_iter / MAX_GEN));
    if isfield(params, 'alm_steps_per_gen')
        ALM_STEPS = max(1, round(params.alm_steps_per_gen));
    end
    
    % --- Initialization ---
    % Population Structure: Gene (Integer 1..M), Lambda (for ALM), Fitness
    Pop = repmat(struct('gene', zeros(1, N), 'lambda', zeros(N, 1), ...
                        'sat', -inf, 'viol', inf), POP_SIZE, 1);
    
    % Pre-calculate Greedy Base for initialization
    if greedy_init_ratio > 0
        try
            raw_greedy = Greedy_Solver_DNN(Task, Fog, Thing, DNN_Data, Pre);
            [~, greedy_gene] = max(raw_greedy, [], 2);
            greedy_gene = greedy_gene';
        catch
            greedy_gene = randi(M, 1, N);
        end
    else
        greedy_gene = randi(M, 1, N);
    end

    greedy_init_count = 0;
    if greedy_init_ratio > 0
        greedy_init_count = max(1, round(POP_SIZE * greedy_init_ratio));
    end
    
    % Initialize Individuals
    for i = 1:POP_SIZE
        if i <= greedy_init_count
            % Greedy / Perturbed Greedy
            Pop(i).gene = greedy_gene;
            % Perturb 10%
            if i > 1 && greedy_perturb_ratio > 0
                 m_idx = randperm(N, max(1, ceil(N * greedy_perturb_ratio)));
                 Pop(i).gene(m_idx) = randi(M, 1, length(m_idx));
            end
        else
            % Random Initialization
            Pop(i).gene = randi(M, 1, N);
        end
        Pop(i).lambda = zeros(N, 1);
    end
    
    % --- Pre-Evaluate Initial Population (CRITICAL for Elitism) ---
    % Evaluate all individuals before the first generation to ensure Global_Best starts correct.
    Global_Best = Pop(1);
    Global_Best.sat = -inf;
    Global_Best.viol = inf;

    for i = 1:POP_SIZE
        % Convert Gene to Binary
        t_bin = zeros(N, M);
        for k = 1:N
            t_bin(k, Pop(i).gene(k)) = 1;
        end
        % Repair & Calculate Metrics (Use same repair as baseline for fairness)
        if use_repair
             t_bin = Perform_Safe_Harbor_Repair(t_bin, Pre, Fog, Task, DNN_Data);
             % Update gene to match repaired binary? 
             % Yes, Lamarckian: phenotype -> genotype
             [~, repaired_gene] = max(t_bin, [], 2);
             Pop(i).gene = repaired_gene';
        end
        
        [sat, viol] = evaluate_for_selection(t_bin, Pre, Task, Fog, DNN_Data, repair_aware_fitness);
        Pop(i).sat = sat;
        Pop(i).viol = viol;
        
        % Update Global Best (Feasibility First Logic)
        % Priority: 1. Feasible > Infeasible
        %           2. If both Feasible: Higher Sat
        %           3. If both Infeasible: Lower Viol
        is_feas_i = (Pop(i).viol <= 1e-6);
        is_feas_best = (Global_Best.viol <= 1e-6);
        
        if (is_feas_i && ~is_feas_best) || ...
           (is_feas_i && is_feas_best && Pop(i).sat > Global_Best.sat) || ...
           (~is_feas_i && ~is_feas_best && Pop(i).viol < Global_Best.viol)
            Global_Best = Pop(i);
        end
    end
    
    loss_history_total = [];

    % --- Main Memetic Loop ---
    for gen = 1:MAX_GEN
        
        % 1. Local Search (Lamarckian Learning)
        % Convert Gene -> Logits -> ALM -> Gene
        % Hybrid Strategy: Only apply ALM to 50% of population to preserve genetic diversity
        % This ensures M-CA-ALM captures the best of both GA and Gradient Descent
        for i = 1:POP_SIZE
            % Probabilistic ALM Application
            if rand() < local_search_prob
                old_gene = Pop(i).gene;
                old_lambda = Pop(i).lambda;
                old_bin = zeros(N, M);
                for k = 1:N, old_bin(k, old_gene(k)) = 1; end
                [old_sat, old_viol] = evaluate_for_selection(old_bin, Pre, Task, Fog, DNN_Data, repair_aware_fitness);

                % A. Gene to Logits (One-Hot-Like)
                % Use +/- 2.0 for soft start
                logits = ones(N, M) * -2.0;
                for k = 1:N
                    logits(k, Pop(i).gene(k)) = 2.0;
                end
                
                % B. Run ALM Gradient Descent
                [t_bin, l_hist, t_logits, t_lambda] = run_alm_core(Task, Fog, Pre, ...
                    logits, Pop(i).lambda, ALM_STEPS, params);
                
                % C. Logits to Gene (Lamarckian Update)
                [~, new_gene] = max(t_logits, [], 2);

                % D. Evaluate Fitness
                % Use the binary output from ALM (which is consistent with t_logits argmax)
                [sat, viol] = evaluate_for_selection(t_bin, Pre, Task, Fog, DNN_Data, repair_aware_fitness);
                if ~guarded_local_search || is_better_candidate(sat, viol, old_sat, old_viol)
                    Pop(i).gene = new_gene';
                    Pop(i).lambda = t_lambda;
                    Pop(i).sat = sat;
                    Pop(i).viol = viol;
                else
                    Pop(i).gene = old_gene;
                    Pop(i).lambda = old_lambda;
                    Pop(i).sat = old_sat;
                    Pop(i).viol = old_viol;
                end
                
                if i == 1
                    loss_history_total = [loss_history_total; l_hist.loss];
                end
            else
                % Pure GA Evaluation (No ALM)
                % Just re-evaluate in case (though static from previous gen, but crossover changes it)
                % Actually, we need to evaluate new individuals from Crossover/Mutation!
                % Wait, Crossover/Mutation happens at the END of the loop.
                % So at the start of the loop, individuals are new.
                % We MUST evaluate them.
                
                theta_bin_ga = zeros(N, M);
                for k=1:N, theta_bin_ga(k, Pop(i).gene(k))=1; end
                
                if use_repair
                    theta_bin_ga = Perform_Safe_Harbor_Repair(theta_bin_ga, Pre, Fog, Task, DNN_Data);
                    [~, repaired_gene] = max(theta_bin_ga, [], 2);
                    Pop(i).gene = repaired_gene';
                end
                
                [sat, viol] = evaluate_for_selection(theta_bin_ga, Pre, Task, Fog, DNN_Data, repair_aware_fitness);
                Pop(i).sat = sat;
                Pop(i).viol = viol;
            end
        end
        
        % 2. Update Global Best (Feasibility First Logic)
        for i = 1:POP_SIZE
             is_feas_i = (Pop(i).viol <= 1e-6);
             is_feas_best = (Global_Best.viol <= 1e-6);
             
             if (is_feas_i && ~is_feas_best) || ...
                (is_feas_i && is_feas_best && Pop(i).sat > Global_Best.sat) || ...
                (~is_feas_i && ~is_feas_best && Pop(i).viol < Global_Best.viol)
                 Global_Best = Pop(i);
             end
        end
        
        % 3. Selection (Tournament with Feasibility Logic)
        New_Pop = Pop;
        for i = 1:POP_SIZE
            c1 = randi(POP_SIZE);
            c2 = randi(POP_SIZE);
            
            p1 = Pop(c1);
            p2 = Pop(c2);
            
            is_feas_1 = (p1.viol <= 1e-6);
            is_feas_2 = (p2.viol <= 1e-6);
            
            better_1 = false;
            if (is_feas_1 && ~is_feas_2)
                better_1 = true;
            elseif (is_feas_1 && is_feas_2 && p1.sat > p2.sat)
                better_1 = true;
            elseif (~is_feas_1 && ~is_feas_2 && p1.viol < p2.viol)
                better_1 = true;
            end
            
            if better_1
                New_Pop(i) = p1;
            else
                New_Pop(i) = p2;
            end
        end
        
        % 4. Crossover (Uniform on Integers)
        for i = 1:2:POP_SIZE-1
            if rand() < 0.8
                Mask = rand(1, N) > 0.5;
                Child1_Gene = New_Pop(i).gene .* Mask + New_Pop(i+1).gene .* (~Mask);
                Child2_Gene = New_Pop(i+1).gene .* Mask + New_Pop(i).gene .* (~Mask);
                New_Pop(i).gene = Child1_Gene;
                New_Pop(i+1).gene = Child2_Gene;
                % Keep parents' lambda or reset? 
                % Keeping is risky after crossover. Let's average or reset. 
                % Resetting is safer for stability.
                New_Pop(i).lambda = zeros(N, 1); 
                New_Pop(i+1).lambda = zeros(N, 1);
            end
        end
        
        % 5. Mutation (Random Resetting)
        for i = 1:POP_SIZE
            if rand() < 0.1
                m_idx = randi(N);
                New_Pop(i).gene(m_idx) = randi(M);
                % Reset lambda if mutated? Maybe.
            end
        end
        
        Pop = New_Pop;
        
        % Elitism
        Pop(1) = Global_Best;
    end
    
    % --- Final Output Extraction ---
    % Convert Gene to Binary Matrix
    theta_bin = zeros(N, M);
    for i = 1:N
        theta_bin(i, Global_Best.gene(i)) = 1;
    end
    
    theta_logits = zeros(N, M); % Dummy return or reconstructed
    for i = 1:N
        theta_logits(i, Global_Best.gene(i)) = 10.0;
    end
    
    loss_hist.loss = loss_history_total;
    loss_hist.sat = [];
    
    % --- Final Repair & Optimization ---
    if use_repair
        % 1. Feasibility Repair (Safe Harbor)
        theta_bin = Perform_Safe_Harbor_Repair(theta_bin, Pre, Fog, Task, DNN_Data);
        
        % 2. Global Optimization (Hill Climbing)
        if enable_internal_global_opt
            theta_bin = Perform_Global_Optimization(theta_bin, Pre, Fog, Task, DNN_Data);
        end
        
    end

end

function tf = is_better_candidate(sat_new, viol_new, sat_old, viol_old)
tol = 1e-12;
new_feasible = viol_new <= tol;
old_feasible = viol_old <= tol;
if new_feasible ~= old_feasible
    tf = new_feasible;
elseif new_feasible
    tf = sat_new > sat_old + tol;
else
    tf = (viol_new < viol_old - tol) || ...
        (abs(viol_new - viol_old) <= tol && sat_new > sat_old + tol);
end
end

function [sat, viol] = evaluate_for_selection(theta, Pre, Task, Fog, DNN_Data, repair_aware)
theta_eval = theta;
if repair_aware
    % Evaluate the phenotype produced by the shared feasibility operator while
    % preserving the unrepaired genotype for population diversity.
    theta_eval = Perform_Safe_Harbor_Repair(theta, Pre, Fog, Task, DNN_Data);
end
[sat, viol] = calculate_metrics_internal(theta_eval, Pre, Task);
end

function [theta_bin, loss_hist, theta_logits, lambda_out] = run_alm_core(Task, Fog, Pre, theta_start, lambda_in, max_iter, params)
    % Core Gradient Descent Loop
    [N, M] = size(theta_start);
    theta_logits = theta_start;
    lambda = lambda_in;
    
    % Hyperparameters
    if isfield(params, 'rho'), rho = params.rho; else, rho = 0.1; end
    if isfield(params, 'mu'), mu = params.mu; else, mu = 1.05; end
    if isfield(params, 'lr'), lr = params.lr; else, lr = 0.05; end
    if isfield(params, 'alpha'), alpha = params.alpha; else, alpha = 1.0; end
    if isfield(params, 'deadline_scale'), d_scale = params.deadline_scale; else, d_scale = 1.0; end
    
    % Adam State (Reset)
    m_t = zeros(N, M);
    v_t = zeros(N, M);
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
    
    loss_hist_vec = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        % 1. Softmax
        theta_prob = softmax_custom(theta_logits')';
        
        % 2. Gradient
        grad = compute_gradients(theta_prob, Pre, lambda, rho, N, M, Task, alpha, d_scale);
        
        % Gradient Clipping
        grad = max(min(grad, 5.0), -5.0);
        
        % 3. Adam Update
        m_t = beta1 * m_t + (1 - beta1) * grad;
        v_t = beta2 * v_t + (1 - beta2) * (grad .^ 2);
        m_hat = m_t / (1 - beta1^iter);
        v_hat = v_t / (1 - beta2^iter);
        theta_logits = theta_logits + lr * m_hat ./ (sqrt(v_hat) + epsilon);
        
        % 4. Penalty Update
        [~, ~, ~, lat_vec, ~, ~] = calculate_metrics_internal(theta_prob, Pre, Task);
        viol = max(0, lat_vec - Task(:, 3) * d_scale);
        lambda = lambda + rho * viol;
        rho = min(rho * mu, 100);
        
        loss_hist_vec(iter) = mean(viol);
    end
    
    lambda_out = lambda;
    history.loss = loss_hist_vec;
    loss_hist = history;
    
    [~, idx] = max(theta_logits, [], 2);
    theta_bin = zeros(N, M);
    for i = 1:N
        theta_bin(i, idx(i)) = 1;
    end
end

function probs = softmax_custom(x)
    ex = exp(x - max(x, [], 1));
    probs = ex ./ sum(ex, 1);
end

function grad = compute_gradients(theta, Pre, lambda, rho_pen, N, M, Task, alpha, d_scale)
    % Exact Sigmoid Satisfaction Gradient
    grad = zeros(N, M);
    
    Mean_Deadline = 1.0;
    if isfield(Pre, 'Mean_Deadline'), Mean_Deadline = Pre.Mean_Deadline; end
    
    Load = sum(theta .* Pre.Comp, 1);
    Rho = Load / Mean_Deadline;
    Rho_Clipped = min(Rho, 0.99);
    
    Wait_Factor = 1.0 + Rho_Clipped ./ (2.0 * (1.0 - Rho_Clipped) + 1e-6);
    Grad_Wait_Factor = 1.0 ./ (2.0 * (1.0 - Rho_Clipped).^2 + 1e-6);
    
    Sum_Theta = sum(theta, 1);
    Ctx_Overhead = 500e-6 * (Sum_Theta .^ 2); 
    
    Lat_Matrix = Pre.Comm + Pre.Comp .* repmat(Wait_Factor, N, 1) + repmat(Ctx_Overhead, N, 1);
    
    Grad_Ctx = 2 * 500e-6 * repmat(Sum_Theta, N, 1);
    
    sigmoid_alpha = alpha;
    
    Exp_Lat = sum(theta .* Lat_Matrix, 2);
    
    Sig_Input_Exp = sigmoid_alpha * (Task(:, 3) * d_scale - Exp_Lat);
    Sat_Exp = 1.0 ./ (1.0 + exp(-Sig_Input_Exp));
    
    Grad_Sat_ExpLat = Sat_Exp .* (1.0 - Sat_Exp) * (-sigmoid_alpha);
    
    Grad_Obj_1 = repmat(Grad_Sat_ExpLat, 1, M) .* Lat_Matrix;
    
    Sens_f = sum(repmat(Grad_Sat_ExpLat, 1, M) .* theta .* Pre.Comp, 1);
    Grad_Obj_2 = (Sens_f .* Grad_Wait_Factor ./ Mean_Deadline) .* Pre.Comp;
    
    Sens_f_Ctx = sum(repmat(Grad_Sat_ExpLat, 1, M) .* theta, 1);
    Grad_Obj_3 = repmat(Sens_f_Ctx, N, 1) .* Grad_Ctx;
    
    grad = Grad_Obj_1 + Grad_Obj_2 + Grad_Obj_3;
    
    Sum_Theta = sum(theta, 2);
    Grad_Pen = 2 * (Sum_Theta - 1);
    
    grad = grad - rho_pen * repmat(Grad_Pen, 1, M);
    
    grad = grad - repmat(lambda, 1, M) .* Lat_Matrix;
end

function [avg_sat, viol_rate, breakdown, Latency_Vec, Sat_Vec, Viol_Vec] = calculate_metrics_internal(theta, Pre, Task)
    [N, M] = size(theta);
    
    Node_Task_Count = sum(theta, 1);
    Ctx_Overhead = 500e-6 * (Node_Task_Count .^ 2);
    
    % Rigorous Rho Calculation (Matches calculate_metrics_v2)
    Mean_Deadline = mean(Task(:, 3));
    if Mean_Deadline < 0.1, Mean_Deadline = 1.0; end
    
    Load_Time = sum(theta .* Pre.Comp, 1);
    Rho = Load_Time / Mean_Deadline;
    
    % Stable Region
    stable_mask = Rho < 0.99;
    Q_Factor = ones(1, M);
    rho_stable = Rho(stable_mask);
    Q_Factor(stable_mask) = 1.0 + rho_stable ./ (2.0 * (1.0 - rho_stable + 1e-6));
    
    % Unstable Region (Penalty matches v2)
    Q_Factor(~stable_mask) = 1e4;
    
    Lat_Matrix = Pre.Comm + Pre.Comp .* repmat(Q_Factor, N, 1) + repmat(Ctx_Overhead, N, 1);
    
    Latency_Vec = sum(theta .* Lat_Matrix, 2);
    
    % Use Binary Satisfaction for Consistent Safety Check
    Sat_Vec = zeros(N, 1);
    for i=1:N
        if Latency_Vec(i) <= Task(i, 3)
            Sat_Vec(i) = 1.0;
        else
            Sat_Vec(i) = 0.0;
        end
    end
    avg_sat = mean(Sat_Vec); % 0.0 to 1.0
    
    Viol_Vec = max(0, Latency_Vec - Task(:, 3));
    viol_rate = mean(Viol_Vec > 0);
    
    breakdown.comm = mean(sum(theta .* Pre.Comm, 2));
    breakdown.comp = mean(sum(theta .* Pre.Comp, 2));
    breakdown.wait = mean(Latency_Vec) - breakdown.comm - breakdown.comp;
end

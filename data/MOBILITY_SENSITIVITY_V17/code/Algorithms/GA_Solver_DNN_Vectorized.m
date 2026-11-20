function [best_theta, best_fitness] = GA_Solver_DNN_Vectorized(Task, Fog, Thing, DNN_Data, pop_size, max_gen, Pre_In, options)
    % GA_Solver_DNN_Vectorized.m
    % High-Performance Genetic Algorithm for Baseline Comparison
    % Uses Vectorized Fitness Evaluation and Greedy Initialization
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    if nargin < 8 || isempty(options)
        options = struct();
    end
    options = normalize_ga_options(options);

    if nargin < 7 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end
    
    % --- 1. Initialization ---
    pop = randi(M, pop_size, N); % Integer representation [1...M]
    
    if options.use_greedy_seed
        try
            theta_greedy = Greedy_Solver_DNN(Task, Fog, Thing, DNN_Data, Pre);
            [~, greedy_indices] = max(theta_greedy, [], 2);
            pop(1, :) = greedy_indices';

            max_seeded = min(pop_size, 1 + ceil(pop_size * options.seed_fraction));
            for k = 2:max_seeded
                pop(k, :) = greedy_indices';
                mut_count = max(1, ceil(N * options.seed_mutation_fraction));
                mut_idx = randperm(N, mut_count);
                pop(k, mut_idx) = randi(M, 1, length(mut_idx));
            end
        catch
        end
    end
    
    best_fitness = -Inf;
    best_theta = zeros(N, M);
    best_sol_idx = pop(1, :);
    
    % --- 2. Evolution Loop ---
    for gen = 1:max_gen
        % Evaluation
        fitness = calculate_fitness_batch(pop, Pre, Task, Fog, Thing, DNN_Data, options);
        
        % Track Best
        [max_fit, idx] = max(fitness);
        if max_fit > best_fitness
            best_fitness = max_fit;
            best_sol_idx = pop(idx, :);
        end
        
        % Elitism: Keep best
        new_pop = zeros(size(pop));
        new_pop(1, :) = best_sol_idx;
        
        % Selection (Tournament) & Crossover
        for i = 2:pop_size
            % Tournament
            p1 = randi(pop_size); p2 = randi(pop_size);
            parent1 = pop(p1, :);
            if fitness(p2) > fitness(p1), parent1 = pop(p2, :); end
            
            p1 = randi(pop_size); p2 = randi(pop_size);
            parent2 = pop(p1, :);
            if fitness(p2) > fitness(p1), parent2 = pop(p2, :); end
            
            % Uniform Crossover
            mask = rand(1, N) > 0.5;
            child = parent1 .* mask + parent2 .* (~mask);
            
            % Mutation
            if rand < 0.1
                m_idx = randi(N);
                child(m_idx) = randi(M);
            end
            
            new_pop(i, :) = child;
        end
        pop = new_pop;
    end
    
    % Convert Best to One-Hot
    best_theta = zeros(N, M);
    for i = 1:N
        best_theta(i, best_sol_idx(i)) = 1;
    end
    
    % Historical callers retain final repair by default. Auditable pipelines can
    % disable it and apply a shared post-processing stage outside every solver.
    if options.apply_final_repair
        best_theta = Perform_Safe_Harbor_Repair(best_theta, Pre, Fog, Task, DNN_Data);
    end
    
end

function options = normalize_ga_options(options)
    if ~isfield(options, 'use_greedy_seed')
        options.use_greedy_seed = true;
    end
    if ~isfield(options, 'seed_fraction')
        options.seed_fraction = 0.2;
    end
    if ~isfield(options, 'seed_mutation_fraction')
        options.seed_mutation_fraction = 0.1;
    end
    if ~isfield(options, 'apply_final_repair')
        options.apply_final_repair = true;
    end
    if ~isfield(options, 'repair_aware_fitness')
        options.repair_aware_fitness = false;
    end
end

function fitness = calculate_fitness_batch(pop, Pre, Task, Fog, Thing, DNN_Data, options)
    % Returns Satisfaction Count (Maximize)
    % UPDATED: Includes M/D/1 Queuing and Context Switching Overhead for FAIR COMPARISON
    
    [pop_size, N] = size(pop);
    [M, ~] = size(Fog);
    fitness = zeros(pop_size, 1);
    
    deadlines = Task(:, 3);
    Mean_Deadline = mean(deadlines);
    if Mean_Deadline < 0.1, Mean_Deadline = 1.0; end
    
    ctx_switch_time = 500e-6; % 500us penalty
    
    for p = 1:pop_size
        assignment = pop(p, :);

        if options.repair_aware_fitness
            % Rank the deployed phenotype but retain the unrepaired integer
            % genotype in pop for selection, crossover, and mutation.
            theta_eval = zeros(N, M);
            theta_eval(sub2ind([N, M], (1:N)', assignment(:))) = 1;
            theta_eval = Perform_Safe_Harbor_Repair(theta_eval, Pre, Fog, Task, DNN_Data);
            [sat, ~] = calculate_metrics_v2(theta_eval, Task, Fog, Thing, DNN_Data, Pre);
            fitness(p) = sat;
            continue;
        end
        
        % 1. Calculate Node Loads (Vectorized where possible)
        % We need sum of Pre.Comp for assigned tasks on each node
        node_base_time = zeros(1, M);
        tasks_per_node = zeros(1, M);
        
        for i = 1:N
            node = assignment(i);
            node_base_time(node) = node_base_time(node) + Pre.Comp(i, node);
            tasks_per_node(node) = tasks_per_node(node) + 1;
        end
        
        % 2. Calculate Rho and Q_Factor (M/D/1)
        % Rho = Arrival / Service. Here approximated by Total_Comp / Window
        Rho = node_base_time / Mean_Deadline;
        Q_Factor = ones(1, M);
        
        % Stable Region
        stable_mask = Rho < 0.99;
        rho_stable = Rho(stable_mask);
        Q_Factor(stable_mask) = 1 + rho_stable ./ (2 * (1 - rho_stable + 1e-6));
        
        % Unstable Region (Soft Penalty for GA evolution)
        % We use a high penalty to guide GA away from instability
        Q_Factor(~stable_mask) = 100.0; 
        
        % 3. Context Switching Overhead
        Ctx_Overhead = ctx_switch_time * (tasks_per_node .^ 2);
        
        % 4. Calculate Satisfaction
        sat_count = 0;
        for i = 1:N
            node = assignment(i);
            
            t_comm = Pre.Comm(i, node);
            t_comp = Pre.Comp(i, node);
            
            % Total Latency = Comm + Comp * Q + Ctx
            total_lat = t_comm + t_comp * Q_Factor(node) + Ctx_Overhead(node);
            
            if total_lat <= deadlines(i)
                sat_count = sat_count + 1;
            end
        end
        
        fitness(p) = sat_count;
    end
end

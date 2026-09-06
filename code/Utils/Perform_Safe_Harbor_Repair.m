function theta_new = Perform_Safe_Harbor_Repair(theta_in, Pre, Fog, Task, DNN_Data)
    % Perform_Safe_Harbor_Repair.m
    % Shared utility for "Safe Harbor" Repair Strategy (G3R V2).
    % Ensures FAIR COMPARISON by allowing Baselines to use the same repair logic.
    %
    % UPDATE: Modified to be a "Soft Repair" that respects the input assignment
    % if it is feasible. This allows the underlying algorithm (ALM/GA) to 
    % influence the result, rather than just running a blind greedy packing.
    
    [N, M] = size(theta_in);
    theta_new = zeros(N, M);
    
    % --- Prepare Memory Data ---
    Task_Mem_MB = zeros(N, 1);
    for i = 1:N
        t_type = Task(i, 2);
        
        % Robust DNN_Data Access
        if isnumeric(DNN_Data)
             % DNN_Data is [Input, Weights, GFLOPs] matrix
             % Mapping: Type 1(VGG)->Row 3, Type 2(ResNet)->Row 2, Type 3(MobileNet)->Row 1
             if t_type == 1, row_idx = 3;
             elseif t_type == 2, row_idx = 2;
             elseif t_type == 3, row_idx = 1;
             else, row_idx = 1; end
             
             % Use Weights (Col 2) + Input (Col 1) as memory approximation
             Task_Mem_MB(i) = DNN_Data(row_idx, 2) + DNN_Data(row_idx, 1);
        elseif iscell(DNN_Data)
            if t_type > length(DNN_Data), t_type = 1; end
            model = DNN_Data{t_type};
            
            % Estimate Memory Usage: Sum of all layer output data (Feature Maps)
            if isfield(model, 'data')
                Task_Mem_MB(i) = sum(model.data); 
            else
                Task_Mem_MB(i) = 100; % Default fallback (100MB)
            end
        elseif isstruct(DNN_Data)
            if t_type > numel(DNN_Data), t_type = 1; end
            model = DNN_Data(t_type);
             
            if isfield(model, 'data')
                Task_Mem_MB(i) = sum(model.data); 
            else
                Task_Mem_MB(i) = 100; % Default fallback (100MB)
            end
        else
            error('DNN_Data must be numeric, cell, or struct array');
        end
    end
    Node_Cap = Fog(:, 8)'; 
    
    % --- Sorting Strategy ---
    % Sort by Min Comp Time Ascending (Easiest First) to maximize Sat Count
    min_comp = min(Pre.Comp, [], 2);
    [~, sort_idx] = sort(min_comp, 'ascend');
    sorted_tasks = sort_idx;
    
    current_mem = zeros(1, M);
    current_load = zeros(1, M);
    current_count = zeros(1, M); % Track task count for context switching
    ctx_switch_time = 500e-6;    % 500us per task interaction
    
    if isfield(Pre, 'Mean_Deadline')
        Mean_Deadline = Pre.Mean_Deadline;
    else
        Mean_Deadline = mean(Task(:, 3));
        if Mean_Deadline < 0.1, Mean_Deadline = 1.0; end
    end
    
    for k = 1:N
        i = sorted_tasks(k);
        
        % Identify Original Choice
        [~, orig_node] = max(theta_in(i, :));
        
        % Strategy: Try Original Node FIRST
        placed = false;
        
        % 1. Try Original Node
        f = orig_node;
        if f > 0 % Ensure valid node
            if current_mem(f) + Task_Mem_MB(i) <= Node_Cap(f)
                new_load = current_load(f) + Pre.Comp(i, f);
                new_rho = new_load / Mean_Deadline;
                
                % Accept if Safe AND Meets Deadline
                if new_rho < 0.95
                    % Calculate Latency (Including Context Switching Overhead)
                    new_count = current_count(f) + 1;
                    ctx_overhead = ctx_switch_time * (new_count ^ 2);
                    
                    q = 1 + new_rho / (2*(1-new_rho));
                    lat = Pre.Comm(i, f) + Pre.Comp(i, f) * q + ctx_overhead;
                    
                    if lat <= Task(i, 3)
                        theta_new(i, f) = 1;
                        current_mem(f) = current_mem(f) + Task_Mem_MB(i);
                        current_load(f) = current_load(f) + Pre.Comp(i, f);
                        current_count(f) = new_count;
                        placed = true;
                    end
                end
            end
        end
        
        % 2. If Original failed, Search for "Safe" Node (Standard G3R)
        if ~placed
            best_node = -1;
            best_lat = inf;
            
            for f = 1:M
                if current_mem(f) + Task_Mem_MB(i) <= Node_Cap(f)
                    new_load = current_load(f) + Pre.Comp(i, f);
                    new_rho = new_load / Mean_Deadline;
                    
                    if new_rho < 0.95
                        new_count = current_count(f) + 1;
                        ctx_overhead = ctx_switch_time * (new_count ^ 2);
                        q = 1 + new_rho / (2*(1-new_rho));
                        lat = Pre.Comm(i, f) + Pre.Comp(i, f) * q + ctx_overhead;
                        
                        if lat < best_lat
                            best_lat = lat;
                            best_node = f;
                        end
                    end
                end
            end
            
            if best_node ~= -1
                theta_new(i, best_node) = 1;
                current_mem(best_node) = current_mem(best_node) + Task_Mem_MB(i);
                current_load(best_node) = current_load(best_node) + Pre.Comp(i, best_node);
                current_count(best_node) = current_count(best_node) + 1;
                placed = true;
            end
        end
        
        % 3. EMERGENCY FALLBACK: If still not placed, DROP the task
        % This prevents "Cascading Failures" where one forced task violates 
        % the entire node and fails all other tasks on it.
        if ~placed
            % Do nothing. theta_new(i, :) remains all zeros (dropped).
        end
    end
end
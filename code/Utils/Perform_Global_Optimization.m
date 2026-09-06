function theta_bin = Perform_Global_Optimization(theta_bin, Pre, Fog, Task, DNN_Data)
    % Perform_Global_Optimization.m
    % Post-Repair Optimization (Hill Climbing) with M/D/1 Awareness
    % Tries to move tasks to faster nodes if capacity allows AND reduces global latency
    
    [N, M] = size(theta_bin);
    
    % 1. Calculate Current Resource Usage & Load
    Task_Mem_MB = zeros(N, 1);
    for i = 1:N
        t_type = Task(i, 2);
        
        % Robust DNN_Data Access
        if isnumeric(DNN_Data)
             % DNN_Data is [Input, Weights, GFLOPs] matrix
             % Mapping: Type 1(VGG)->Row 3, Type 2(ResNet)->Row 2, Type 3(MobileNet)->Row 1
             if t_type == 1
                 row_idx = 3;
             elseif t_type == 2
                 row_idx = 2;
             elseif t_type == 3
                 row_idx = 1;
             else
                 row_idx = 1;
             end
             
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
    
    % Check Pre fields
    if ~isfield(Pre, 'Comp') || ~isfield(Pre, 'Comm')
        return; % Cannot optimize
    end
    
    % Initial Node State
    Node_Mem = zeros(1, M);
    Node_Load_Time = zeros(1, M); % Sum of T_cmp
    
    for i = 1:N
        node = find(theta_bin(i, :) == 1);
        if isempty(node), continue; end
        
        Node_Mem(node) = Node_Mem(node) + Task_Mem_MB(i);
        Node_Load_Time(node) = Node_Load_Time(node) + Pre.Comp(i, node);
    end
    
    Memory_Cap = get_memory_caps(Fog, M);
    
    % M/D/1 Parameters
    Ctx_Switch_Time = 500e-6; % 500us
    
    % 2. Hill Climbing with M/D/1 Queuing
    % Iterate multiple times to propagate improvements
    MAX_PASSES = 3;
    
    % DEBUG: Track improvement
    % initial_lat = sum(sum(theta_bin .* (Pre.Comm + Pre.Comp))); 
    % fprintf('    [HC Start] Latency Estimate: %.4f\n', initial_lat);
    
    for pass = 1:MAX_PASSES
        improved = false;
        moved_count = 0;
        
        % Randomize order to avoid bias
        indices = randperm(N);
        
        for k = 1:N
            i = indices(k);
            curr_node = find(theta_bin(i, :) == 1);
            if isempty(curr_node), continue; end
            
            % --- 1. Calculate Current Cost (Latency) ---
            % Mean Deadline for Rho normalization
            mean_deadline = mean(Task(:, 3));
            if mean_deadline < 1e-6, mean_deadline = 1.0; end
            
            % Current Node Metrics
            curr_load = Node_Load_Time(curr_node);
            curr_rho = curr_load / mean_deadline;
            
            % M/D/1 Wait Factor (Continuous Extension)
            if curr_rho < 0.999
                 curr_wait = 1.0 + curr_rho / (2.0 * (1.0 - curr_rho));
            else
                 % Linear Penalty for Overload to provide gradient
                 % At rho=0.999, wait ~ 500.5
                 % For rho > 1, slope = 1000
                 curr_wait = 500.5 + (curr_rho - 0.999) * 1000.0;
            end
            
            % Context Switching
            curr_count = sum(theta_bin(:, curr_node));
            curr_ctx = Ctx_Switch_Time * (curr_count^2);
            
            curr_lat = Pre.Comm(i, curr_node) + Pre.Comp(i, curr_node) * curr_wait + curr_ctx;
            deadline = Task(i, 3);
            
            best_node = curr_node;
            found_better = false;
            
            % --- 2. Try to find a better node ---
            for j = 1:M
                if j == curr_node, continue; end
                
                % Check Memory
                if Node_Mem(j) + Task_Mem_MB(i) > Memory_Cap(j)
                    continue;
                end
                
                % Check Target Latency
                new_load_j = Node_Load_Time(j) + Pre.Comp(i, j);
                new_rho_j = new_load_j / mean_deadline;
                if new_rho_j >= 0.999
                    continue;
                end
                
                if new_rho_j < 0.999
                     new_wait_j = 1.0 + new_rho_j / (2.0 * (1.0 - new_rho_j));
                else
                     new_wait_j = 500.5 + (new_rho_j - 0.999) * 1000.0;
                end
                
                % Approximate new context switch (ignoring that we removed task i from curr_node)
                % Actually, context switch on target node increases
                new_count_j = sum(theta_bin(:, j)) + 1;
                new_ctx_j = Ctx_Switch_Time * (new_count_j^2);
                
                cand_lat = Pre.Comm(i, j) + Pre.Comp(i, j) * new_wait_j + new_ctx_j;
                
                % Decision Logic: Maximize Satisfaction, then Minimize Latency
                curr_viol = max(0, curr_lat - deadline);
                cand_viol = max(0, cand_lat - deadline);
                
                is_better = false;
                
                if curr_viol > 0
                    if cand_viol < curr_viol
                        is_better = true; % Reduced violation
                    end
                elseif cand_viol == 0
                    if cand_lat < curr_lat
                        is_better = true; % Improved latency (both valid)
                    end
                end
                
                if is_better
                    % --- CASCADING FAILURE PREVENTION (CRITICAL) ---
                    % Verify that moving task i to j does not cause existing tasks on j to miss their deadlines.
                    % This prevents "robbing Peter to pay Paul" where one move kills multiple existing tasks.
                    
                    safe_move = true;
                    
                    % Only check if target node is under pressure
                    if new_rho_j > 0.8
                        tasks_on_j = find(theta_bin(:, j));
                        for t_idx = tasks_on_j'
                             % Calculate new latency for task t_idx
                             % Latency = Comm + Comp * new_wait + new_ctx
                             t_lat_new = Pre.Comm(t_idx, j) + Pre.Comp(t_idx, j) * new_wait_j + new_ctx_j;
                             
                             if t_lat_new > Task(t_idx, 3) % Violates deadline
                                 % Check if it was ALREADY violating
                                 % Calculate old latency (needs old wait)
                                 old_load_j = Node_Load_Time(j); % Before adding task i
                                 old_rho_j = old_load_j / mean_deadline;
                                 if old_rho_j < 0.999
                                     old_wait_j = 1.0 + old_rho_j / (2.0 * (1.0 - old_rho_j));
                                 else
                                     old_wait_j = 500.5 + (old_rho_j - 0.999) * 1000.0;
                                 end
                                 
                                 old_count_j = sum(theta_bin(:, j));
                                 old_ctx_j = Ctx_Switch_Time * (old_count_j^2);
                                 
                                 t_lat_old = Pre.Comm(t_idx, j) + Pre.Comp(t_idx, j) * old_wait_j + old_ctx_j;
                                 
                                 if t_lat_old <= Task(t_idx, 3)
                                     % It was satisfied, now violated -> CASCADING FAILURE
                                     safe_move = false;
                                     break; 
                                 end
                             end
                        end
                    end
                    
                    if safe_move
                        best_node = j;
                        found_better = true;
                        break; % First Fit (Greedy)
                    end
                end
            end
            
            if found_better
                % Apply Move
                theta_bin(i, curr_node) = 0;
                theta_bin(i, best_node) = 1;
                
                % Update State
                Node_Mem(curr_node) = Node_Mem(curr_node) - Task_Mem_MB(i);
                Node_Mem(best_node) = Node_Mem(best_node) + Task_Mem_MB(i);
                
                Node_Load_Time(curr_node) = Node_Load_Time(curr_node) - Pre.Comp(i, curr_node);
                Node_Load_Time(best_node) = Node_Load_Time(best_node) + Pre.Comp(i, best_node);
                
                improved = true;
                moved_count = moved_count + 1;
            end
        end
        
        % fprintf('    [HC Pass %d] Moved: %d tasks\n', pass, moved_count);
        if ~improved, break; end
    end
end

function caps = get_memory_caps(Fog, M)
    if size(Fog, 2) >= 8
        caps = Fog(:, 8)';
    else
        caps = Fog(:, end)';
    end
    if numel(caps) < M
        caps = [caps, inf(1, M - numel(caps))];
    elseif numel(caps) > M
        caps = caps(1:M);
    end
end

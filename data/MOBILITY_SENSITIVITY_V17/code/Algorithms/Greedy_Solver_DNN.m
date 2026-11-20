function theta = Greedy_Solver_DNN(Task, Fog, Thing, DNN_Data, Pre_In)
    % Greedy_Solver_DNN.m
    % Deadline-Aware Greedy Heuristic for Heterogeneous Fog
    % Assigns task to the node that offers Minimum Latency (taking into account current load)
    % Prioritizes tasks with tighter deadlines.
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    % --- 1. Pre-computation ---
    if nargin < 5 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end
    
    % --- 2. Sort Tasks by Deadline Tightness ---
    % Tighter deadline -> Higher priority
    deadlines = Task(:, 3);
    [~, task_order] = sort(deadlines, 'ascend');
    
    % --- 3. Initialize System State ---
    theta = zeros(N, M);
    Load_Time = zeros(M, 1); % Current accumulated execution time on each node
    
    % --- 4. Greedy Assignment ---
    for i = 1:N
        task_idx = task_order(i);
        best_node = -1;
        min_est_latency = Inf;
        
        for f = 1:M
            % Base Latency (Comm + Comp)
            t_comm = Pre.Comm(task_idx, f);
            t_comp = Pre.Comp(task_idx, f);
            
            % Wait Time Estimation
            % T_wait = Current_Load_Time
            t_wait = Load_Time(f);
            
            % Overload Protection (Enhanced Baseline)
            % Check if adding this task would exceed stable utilization (rho > 0.999)
            % Approximate Rho = (Load_Time + t_comp) / Mean_Deadline
            mean_deadline = mean(Task(:, 3));
            if mean_deadline < 1e-6, mean_deadline = 1.0; end
            
            current_rho = (Load_Time(f) + t_comp) / mean_deadline;
            
            if current_rho > 0.999
                % Node is overloaded, skip it unless it's the only option
                est_latency = Inf; 
            else
                % Context Switch Overhead (Approximation)
                % Simple linear penalty for greedy
                t_ctx = 0; 
                
                est_latency = t_comm + t_comp + t_wait + t_ctx;
            end
            
            if est_latency < min_est_latency
                min_est_latency = est_latency;
                best_node = f;
            end
        end
        
        % Fallback Strategy: If all nodes are overloaded (min_est_latency == Inf)
        if best_node == -1
            % Strategy: Pick the node with minimum current load (Rho) to minimize damage
            min_rho = Inf;
            best_node = 1; % Default to 1
            
            for f = 1:M
                curr_rho = Load_Time(f) / mean_deadline;
                if curr_rho < min_rho
                    min_rho = curr_rho;
                    best_node = f;
                end
            end
            % Note: This task will likely violate deadline, but we must assign it.
        end
        
        % Assign
        theta(task_idx, best_node) = 1;
        
        % Update Load
        % Add this task's computation time to the node's load
        Load_Time(best_node) = Load_Time(best_node) + Pre.Comp(task_idx, best_node);
    end
    
end

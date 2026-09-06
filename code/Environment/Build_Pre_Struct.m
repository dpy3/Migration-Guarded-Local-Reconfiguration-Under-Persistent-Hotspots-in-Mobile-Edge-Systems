function Pre = Build_Pre_Struct(Task, Fog, Thing, DNN_Data)
    % Build_Pre_Struct.m
    % Pre-calculates computation and communication matrices for efficiency.
    %
    % Inputs:
    %   Task, Fog, Thing, DNN_Data (from Generate_Environment)
    %
    % Outputs:
    %   Pre struct with:
    %     .Comp (N x M): Execution time (s) - Initial Estimate (Overwritten by Trace)
    %     .Comm (N x M): Transmission time (s)
    %     .Node_Types (1 x M): Fog node types
    %     .Cap (1 x M): Fog node capacities
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    Pre.Comp = zeros(N, M);
    Pre.Comm = zeros(N, M);
    % Fog(:, 3) is overloaded in some contexts, but we trust the caller to handle Node_Types
    % if strictly needed. Here we just init.
    Pre.Cap = Fog(:, 1)';
    
    % --- 1. Computation Time (Theoretical Fallback) ---
    % T_comp = Workload (GFLOPs) / Capacity (GHz)
    for i = 1:N
        workload = Task(i, 2); % Note: Task(:,2) is Type in Env, but here treated as Workload?
        % Actually in Generate_Environment, Task(:,5) is Workload (GFLOPs).
        % Task(:,2) is Type ID.
        % Let's fix this logic to use Column 5 if available.
        
        if size(Task, 2) >= 5
             workload = Task(i, 5);
        else
             workload = 1.0; % Default
        end
        
        for j = 1:M
            cap = Fog(j, 1);
            if cap <= 0, cap = 0.1; end
            Pre.Comp(i, j) = workload / cap;
        end
    end
    
    % --- 2. Communication Time ---
    % Shannon's Formula: R = B * log2(1 + P*h / N0)
    
    B = 20e6; % 20 MHz Bandwidth
    P_tx = 0.1; % 100 mW Transmission Power
    N0_dBm = -174 + 10*log10(B);
    N0 = 10^(N0_dBm / 10) / 1000; % Watts
    
    % Fog Node Locations
    % Identify Cloud (Capacity > 1000)
    cloud_idx = find(Fog(:, 4) > 1000);
    if isempty(cloud_idx), cloud_idx = M; end 
    
    Fog_Loc = zeros(M, 2);
    % Cloud Location (Remote)
    Fog_Loc(cloud_idx, :) = [250, 250]; 
    
    % Edges (Grid Distribution)
    grid_size = ceil(sqrt(M-1));
    if grid_size < 1, grid_size = 1; end
    step = 500 / grid_size;
    idx = 1;
    for x = step/2 : step : 500
        for y = step/2 : step : 500
            if idx > M, break; end
            if idx == cloud_idx
                idx = idx + 1; 
                if idx > M, break; end
            end
            Fog_Loc(idx, :) = [x, y];
            idx = idx + 1;
        end
    end
    
    for i = 1:N
        user_loc = Thing(i, 1:2);
        % Use Task Col 4 (Input Size MB)
        if size(Task, 2) >= 4
            data_size_bits = Task(i, 4) * 8 * 1e6; 
        else
            data_size_bits = 1.0 * 8 * 1e6;
        end
        
        for j = 1:M
            dist = norm(user_loc - Fog_Loc(j, :));
            if dist < 1.0, dist = 1.0; end 
            
            % Path Loss Model (Urban Macro)
            d_km = dist / 1000;
            pl_db = 128.1 + 37.6 * log10(d_km);
            path_loss = 10^(-pl_db / 10);
            
            h = path_loss; 
            snr = (P_tx * h) / N0;
            rate = B * log2(1 + snr);
            
            % Cloud Backhaul
            if j == cloud_idx
                rate = 100e6; % 100 Mbps
            end
            
            Pre.Comm(i, j) = data_size_bits / rate;
            
            % Cloud RTT Penalty
            if j == cloud_idx
                Pre.Comm(i, j) = Pre.Comm(i, j) + 0.050; % 50ms RTT
            end
        end
    end
end

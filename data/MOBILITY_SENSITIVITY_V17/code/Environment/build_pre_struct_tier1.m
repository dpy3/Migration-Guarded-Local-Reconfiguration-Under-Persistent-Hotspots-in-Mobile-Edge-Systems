function Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data)
    % build_pre_struct_tier1.m
    % Pre-calculates computation and communication matrices for efficiency.
    % Compatible with generate_tier1_env_hetero.m
    %
    % Inputs:
    %   Task: [N x 5] (Col 2: Type, Col 3: Deadline)
    %   Fog: [M x 8] (Col 1-2: Loc, Col 4: Cap)
    %   Thing: [N x 6] (Col 1-2: Loc)
    %   DNN_Data: {vgg16, resnet, mobilenet} structs
    %
    % Outputs:
    %   Pre struct with:
    %     .Comp (N x M): Execution time (s)
    %     .Comm (N x M): Transmission time (s)
    %     .Node_Types (1 x M): Fog node types (Derived from Cap/Eff)
    %     .Cap (1 x M): Fog node capacities
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    Pre.Comp = zeros(N, M);
    Pre.Comm = zeros(N, M);
    Pre.Node_Types = zeros(1, M);
    cap = Fog(:, 4);
    c_max = max(cap);
    c_min = min(cap);
    for j = 1:M
        if cap(j) >= max(1000, 0.5 * c_max) && c_max >= 10 * max(c_min, 1e-6)
            Pre.Node_Types(j) = 3;
        elseif cap(j) >= median(cap)
            Pre.Node_Types(j) = 1;
        else
            Pre.Node_Types(j) = 2;
        end
    end
    
    Pre.Cap = Fog(:, 4)';
    Pre.Deadlines = Task(:, 3);
    Pre.Mean_Deadline = max(mean(Task(:, 3)), 1e-3);
    Pre.Violation_Mode = 'resource';
    
    % --- 1. Computation Time ---
    % T_comp = Workload (GFLOPs) / Capacity (GHz)
    for i = 1:N
        type_idx = Task(i, 2);
        
        if isnumeric(DNN_Data)
            % DNN_Data is [Input, Weights, GFLOPs] matrix
            % Mapping: Type 1(VGG)->Row 3, Type 2(ResNet)->Row 2, Type 3(MobileNet)->Row 1
            if type_idx == 1, row_idx = 3;
            elseif type_idx == 2, row_idx = 2;
            elseif type_idx == 3, row_idx = 1;
            else, row_idx = 1; end
            
            total_workload = DNN_Data(row_idx, 3); % Col 3 is GFLOPs
        elseif iscell(DNN_Data)
            % DNN_Data is a cell array {vgg16, resnet, mobilenet}
            % Type 1=VGG, 2=ResNet, 3=MobileNet
            if type_idx >= 1 && type_idx <= length(DNN_Data)
                model = DNN_Data{type_idx};
                % Total Workload (Sum of all layers)
                total_workload = sum(model.comp);
            else
                total_workload = 1.0; % Fallback
            end
        else
            total_workload = 1.0;
        end
        
        for j = 1:M
            cap = Fog(j, 4);
            if cap < 1e-6, cap = 1e-6; end
            Pre.Comp(i, j) = total_workload / cap;
        end
    end
    
    % --- 2. Communication Time ---
    % Shannon's Formula: R = B * log2(1 + P*h / N0)
    % B = 20MHz/40MHz depending on node type
    
    P_tx = 0.1; % 100 mW Transmission Power (Thing)
    N0_dBm = -174 + 10*log10(20e6);
    N0 = 10^(N0_dBm / 10) / 1000; % Watts
    
    for i = 1:N
        user_loc = Thing(i, 1:2);
        
        % Data Size (Input Layer)
        type_idx = Task(i, 2);
        
        if isnumeric(DNN_Data)
            % Mapping: Type 1(VGG)->Row 3, Type 2(ResNet)->Row 2, Type 3(MobileNet)->Row 1
            if type_idx == 1, row_idx = 3;
            elseif type_idx == 2, row_idx = 2;
            elseif type_idx == 3, row_idx = 1;
            else, row_idx = 1; end
            
            data_size_mb = DNN_Data(row_idx, 1); % Col 1 is Input MB
        elseif iscell(DNN_Data)
            if type_idx >= 1 && type_idx <= length(DNN_Data)
                model = DNN_Data{type_idx};
                % Assuming first element of data vector is input size
                data_size_mb = model.data(1); 
            else
                data_size_mb = 1.0;
            end
        else
            data_size_mb = 1.0;
        end
        
        data_bits = data_size_mb * 8 * 1e6;
        
        for j = 1:M
            fog_loc = Fog(j, 1:2);
            dist = norm(user_loc - fog_loc);
            if dist < 1, dist = 1; end
            
            % Channel Gain (Path Loss Model)
            % PL(d) = 128.1 + 37.6 log10(d_km)
            d_km = dist / 1000;
            pl_db = 128.1 + 37.6 * log10(d_km);
            h = 10^(-pl_db / 10);
            
            % Bandwidth
            bw = Fog(j, 6); % Column 6 is BW
            if bw == 0, bw = 20e6; end
            
            % Rate
            snr = P_tx * h / N0;
            rate = bw * log2(1 + snr);
            
            Pre.Comm(i, j) = data_bits / rate;
        end
    end
    
end

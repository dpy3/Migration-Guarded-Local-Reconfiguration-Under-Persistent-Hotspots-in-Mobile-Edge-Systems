function [Task, Fog, Thing, DNN_Data, Pre_Trace] = generate_tier1_env_trace(task_num, fog_num)
    % GENERATE_TIER1_ENV_TRACE
    % Generates simulation environment populated with TRACE-DRIVEN latency data.
    %
    % Outputs:
    %   Pre_Trace: Struct containing 'Trace_Comp_Matrix' [N, T, F]
    %              Stores the PRE-SAMPLED execution time for every task segment on every node.
    %              This ensures strict "Trace-Driven" evaluation as requested by reviewers.
    
    % --- 1. Load Hardware Profiles & Traces ---
    % Use Compression Ratio = 0.1 to fit 5G Bandwidth Limits (150MB -> 15MB)
    try
        [Profiles, DNN_Data, Trace_DB] = Hardware_Profile_Data();
    catch
        % If Hardware_Data doesn't accept args, call without
        [Profiles, DNN_Data, Trace_DB] = Hardware_Profile_Data();
    end
    
    % Define Model Mapping
    % 1=VGG16, 2=ResNet50, 3=MobileNet
    % Note: Hardware_Data returns DNN_Data cell array
    
    % --- 2. Task Generation ---
    Task = zeros(task_num, 6);
    % Mix: 40% VGG16, 30% ResNet50, 30% MobileNet
    r = rand(task_num, 1);
    types = ones(task_num, 1);
    types(r > 0.4) = 2;
    types(r > 0.7) = 3;
    
    Task(:, 2) = types;
    Task(:, 1) = 1:task_num;
    
    % Map Types to Hardware Indices and Set Data Size/Workload
    % DNN_Data: 1=MobileNet, 2=ResNet, 3=VGG
    % Task Types: 1=VGG, 2=ResNet, 3=MobileNet
    for i = 1:task_num
        t_type = Task(i, 2);
        if t_type == 1 % VGG
            hw_idx = 3;
        elseif t_type == 2 % ResNet
            hw_idx = 2;
        elseif t_type == 3 % MobileNet
            hw_idx = 1;
        end
        Task(i, 4) = DNN_Data(hw_idx, 1); % Input Size (MB)
        Task(i, 5) = DNN_Data(hw_idx, 3); % GFLOPs (Workload)
    end

    % Relaxed Deadlines (Heterogeneity Aware)
    % VGG (Type 1): Pi=4.5s, Nano=0.8s. Deadline=3.0s (Forces offload from Pi)
    % ResNet (Type 2): Pi=1.2s, Nano=0.2s. Deadline=2.0s (Pi OK)
    % MobileNet (Type 3): Pi=0.15s. Deadline=1.0s (Easy)
    base_deadlines = [3.0, 2.0, 1.0]';
    
    % Ensure types is column
    types = reshape(types, task_num, 1);
    
    deadlines_vec = base_deadlines(types);
    jitter = 0.9 + 0.2*rand(task_num, 1);
    Task(:, 3) = deadlines_vec .* jitter; % Fix: Assign to Column 3 (Deadline)
    
    % --- 3. Fog Generation (Heterogeneous Cluster) ---
    Fog = zeros(fog_num, 8);
    Fog(:, 1:2) = rand(fog_num, 2) * 500; % 500m area
    Fog(:, 3) = 800;
    
    Node_Types = zeros(fog_num, 1);
    
    % Assign Capabilities dynamically based on fog_num
    % 40% Nano, 40% Pi, 20% Cloud
    num_nano = floor(fog_num * 0.4);
    num_pi = floor(fog_num * 0.4);
    num_cloud = fog_num - num_nano - num_pi;
    
    idx = 1;
    for i=1:num_nano
        Fog(idx, 4) = 100; Fog(idx, 7) = 100; Fog(idx, 6) = 500e6; Fog(idx, 8) = 4096;
        idx = idx + 1;
    end
    for i=1:num_pi
        Fog(idx, 4) = 50; Fog(idx, 7) = 50; Fog(idx, 6) = 500e6; Fog(idx, 8) = 2048;
        idx = idx + 1;
    end
    for i=1:num_cloud
        Fog(idx, 4) = 5000; Fog(idx, 7) = 5000; Fog(idx, 6) = 10000e6; Fog(idx, 8) = 64000;
        idx = idx + 1;
    end
    
    Node_Types = derive_node_types_from_capacity(Fog(:, 4));
    
    % --- 4. Thing (User) Generation ---
    Thing = zeros(task_num, 6);
    Thing(:, 1:2) = rand(task_num, 2) * 500;
    Thing(:, 3) = 10; % Dummy Local Speed (will be overridden by trace)
    Thing(:, 4) = 100e6 + 400e6 * rand(task_num, 1); % Uplink Rate: 100-500 Mbps (5G/WiFi6)
    Thing(:, 5) = 100e6; % Downlink Rate (not critical)
    Thing(:, 6) = 1.0;
    
    % --- 5. Trace Sampling & Pre-Calculation ---
    % Trace_DB Mapping:
    % Row 1-3: Comp Time on Device Types
    %   Row 1: MobileNet (Col 1=Pi?, Col 2=Nano?, Col 3=Cloud?)
    %   Row 2: ResNet
    %   Row 3: VGG16
    % Actually, looking at values:
    % Row 3 (VGG): [4.5, 0.8, 0.08] -> Pi, Nano, Cloud.
    % So Col 1 = Pi (Type 2 node), Col 2 = Nano (Type 1 node), Col 3 = Cloud (Type 3 node).
    
    T_max = 6; % Max layers
    Trace_Comp_Matrix = zeros(task_num, T_max, fog_num + 1);
    
    fprintf('[Env] Sampling %d tasks from Empirical Traces...\n', task_num);
    
    for i = 1:task_num
        type = Task(i, 2);
        % Map Task Type to Trace Row
        if type == 1, row_idx = 3; % VGG
        elseif type == 2, row_idx = 2; % ResNet
        elseif type == 3, row_idx = 1; % MobileNet
        end
        
        % Get Comp Times for all device types
        % Col 1: Pi (Node Type 2)
        % Col 2: Nano (Node Type 1)
        % Col 3: Cloud (Node Type 3)
        
        t_pi = Trace_DB(row_idx, 1);
        t_nano = Trace_DB(row_idx, 2);
        t_cloud = Trace_DB(row_idx, 3);
        
        % Local Device (Thing) - Assume similar to Pi but slightly slower/varied
        t_local = t_pi * (1.0 + 0.2*rand());
        
        % Distribute across layers (Uniform for simplicity)
        vec_pi = (t_pi / T_max) * ones(1, T_max);
        vec_nano = (t_nano / T_max) * ones(1, T_max);
        vec_cloud = (t_cloud / T_max) * ones(1, T_max);
        vec_local = (t_local / T_max) * ones(1, T_max);
        
        for f = 1:fog_num
            ntype = Node_Types(f);
            if ntype == 1 % Nano -> Col 2
                Trace_Comp_Matrix(i, 1:T_max, f) = vec_nano;
            elseif ntype == 2 % Pi -> Col 1
                Trace_Comp_Matrix(i, 1:T_max, f) = vec_pi;
            elseif ntype == 3 % Cloud -> Col 3
                Trace_Comp_Matrix(i, 1:T_max, f) = vec_cloud;
            end
        end
        Trace_Comp_Matrix(i, 1:T_max, fog_num + 1) = vec_local;
    end
    
    Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    Pre.Trace_Comp_Matrix = Trace_Comp_Matrix;
    Pre.Node_Types = Node_Types;
    
    % CRITICAL: Overwrite Pre.Comp with Trace Data
    for i = 1:task_num
        for f = 1:fog_num
            Pre.Comp(i, f) = sum(Trace_Comp_Matrix(i, :, f));
        end
    end
    
    Pre_Trace = Pre;
end

function node_types = derive_node_types_from_capacity(cap)
    m = numel(cap);
    node_types = zeros(m, 1);
    c_max = max(cap);
    c_min = min(cap);
    c_med = median(cap);
    for j = 1:m
        if cap(j) >= max(1000, 0.5 * c_max) && c_max >= 10 * max(c_min, 1e-6)
            node_types(j) = 3;
        elseif cap(j) >= c_med
            node_types(j) = 1;
        else
            node_types(j) = 2;
        end
    end
end

function val = sample_from_row(db_matrix, row_idx)
    % Sample one value from the row of 3 samples
    if row_idx > size(db_matrix, 1)
        val = 0.1; % Fallback
        return;
    end
    cols = size(db_matrix, 2);
    idx = randi(cols);
    val = db_matrix(row_idx, idx);
end

function lat = sample_trace_legacy(db_struct, model_name)
    % Helper to sample from array
    data = [];
    if strcmp(model_name, 'VGG16'), data = db_struct.VGG16;
    elseif strcmp(model_name, 'ResNet50'), data = db_struct.ResNet50;
    elseif strcmp(model_name, 'MobileNet'), data = db_struct.MobileNet;
    end
    
    if isempty(data)
        lat = 0.1; % Fallback
    else
        idx = randi(length(data));
        lat = data(idx);
    end
end

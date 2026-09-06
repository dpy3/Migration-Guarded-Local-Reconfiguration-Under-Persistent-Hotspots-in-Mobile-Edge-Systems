function [Task, Fog, Thing, DNN_Data, Pre_Trace] = Generate_Environment(task_num, fog_num)
    % Generate_Environment.m
    % Generates simulation environment populated with TRACE-DRIVEN latency data.
    % Ensures physical realism and reproducibility.
    %
    % Outputs:
    %   Pre_Trace: Struct containing 'Trace_Comp_Matrix' [N, T, F]
    %              Stores the PRE-SAMPLED execution time for every task segment on every node.
    
    % --- 1. Load Hardware Profiles & Traces ---
    try
        [Profiles, DNN_Data, Trace_DB] = Hardware_Profile_Data();
    catch
        error('Hardware_Profile_Data.m not found or invalid.');
    end
    
    % --- 2. Task Generation ---
    Task = zeros(task_num, 6);
    % Mix: 40% VGG16, 30% ResNet50, 30% MobileNet
    r = rand(task_num, 1);
    types = ones(task_num, 1);
    types(r > 0.4) = 2;
    types(r > 0.7) = 3;
    
    Task(:, 2) = types;
    Task(:, 1) = 1:task_num;
    
    % Map Types to Hardware Indices
    % Task Types: 1=VGG, 2=ResNet, 3=MobileNet
    % DNN_Data: 1=MobileNet, 2=ResNet, 3=VGG (Note the index swap!)
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
    
    types = reshape(types, task_num, 1);
    deadlines_vec = base_deadlines(types);
    
    % Add Jitter (0.9x to 1.1x)
    jitter = 0.9 + 0.2*rand(task_num, 1);
    Task(:, 3) = deadlines_vec .* jitter; 
    
    % --- 3. Fog Generation (Heterogeneous Cluster) ---
    Fog = zeros(fog_num, 8);
    Fog(:, 1:2) = rand(fog_num, 2) * 500; % 500m area
    Fog(:, 3) = 800; % Range (visual only)
    
    % Node Types for Trace Mapping
    % 1=Nano, 2=Pi, 3=Cloud
    Node_Types = zeros(fog_num, 1);
    
    % Assign Capabilities
    % Fog 1-2: Nano (GPU) - 100 GFLOPS
    Fog(1:2, 4) = 100; Fog(1:2, 7) = 100; Fog(1:2, 6) = 500e6; Fog(1:2, 8) = 4096;
    Node_Types(1:2) = 1;

    % Fog 3-4: Pi (CPU) - 50 GFLOPS
    Fog(3:4, 4) = 50; Fog(3:4, 7) = 50; Fog(3:4, 6) = 500e6; Fog(3:4, 8) = 2048;
    Node_Types(3:4) = 2;
    
    % Cloud (5000 GFLOPS)
    Fog(5, 4) = 5000; Fog(5, 7) = 5000; Fog(5, 6) = 10000e6; Fog(5, 8) = 64000;
    Node_Types(5) = 3;
    
    % --- 4. Thing (User) Generation ---
    Thing = zeros(task_num, 6);
    Thing(:, 1:2) = rand(task_num, 2) * 500;
    Thing(:, 3) = 10; % Dummy Local Speed
    Thing(:, 4) = 100e6 + 400e6 * rand(task_num, 1); % Uplink Rate
    Thing(:, 5) = 100e6; 
    Thing(:, 6) = 1.0;
    
    % --- 5. Trace Sampling & Pre-Calculation ---
    T_max = 6; % Max layers
    Trace_Comp_Matrix = zeros(task_num, T_max, fog_num + 1);
    
    % fprintf('[Env] Sampling %d tasks from Empirical Traces...\n', task_num);
    
    for i = 1:task_num
        type = Task(i, 2);
        % Map Task Type to Trace Row
        if type == 1, row_idx = 3; % VGG
        elseif type == 2, row_idx = 2; % ResNet
        elseif type == 3, row_idx = 1; % MobileNet
        end
        
        % Trace_DB Columns: [Pi, Nano, Cloud]
        % Node Type Map: 1=Nano, 2=Pi, 3=Cloud
        % So: Nano->Col 2, Pi->Col 1, Cloud->Col 3
        
        t_pi = Trace_DB(row_idx, 1);
        t_nano = Trace_DB(row_idx, 2);
        t_cloud = Trace_DB(row_idx, 3);
        
        t_local = t_pi * (1.0 + 0.2*rand());
        
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
    
    % Build Pre Structure
    Pre = Build_Pre_Struct(Task, Fog, Thing, DNN_Data);
    Pre.Trace_Comp_Matrix = Trace_Comp_Matrix;
    Pre.Node_Types = Node_Types;
    
    % CRITICAL: Overwrite Pre.Comp with Trace Data for Validity
    for i = 1:task_num
        for f = 1:fog_num
            Pre.Comp(i, f) = sum(Trace_Comp_Matrix(i, :, f));
        end
    end
    
    Pre_Trace = Pre;
end

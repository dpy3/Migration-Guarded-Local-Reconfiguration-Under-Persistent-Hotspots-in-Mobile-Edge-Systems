function [Task, Fog, Thing, DNN_Data, Pre] = generate_tier1_env_hetero(task_num, fog_num)
    % GENERATE_TIER1_ENV_HETERO
    % Generates a heterogeneous, resource-constrained batch offloading scenario.
    %
    % Scenario:
    % - Fog nodes: mixed compute, bandwidth, and memory capacities
    % - Tasks: mixed VGG16, ResNet50, and MobileNetV2 requests
    % - Resource constraints: intentionally tight to study congestion handling
    
    % --- 1. Load Hardware Profiles ---
    Profiles = load_hardware_profile();
    
    % Data scaling for batch mobile inference
    Data_Scale = 0.05; 
    Comp_Scale = 1.2;
    
    vgg16 = Profiles.vgg16;
    vgg16.data = vgg16.data * Data_Scale;
    vgg16.comp = vgg16.comp * Comp_Scale;
    
    % Enforce T=6 (VGG16 standard) for all profiles to allow matrix operations
    target_T = length(vgg16.data);
    
    % Fallback if ResNet/MobileNet missing
    if isfield(Profiles, 'resnet')
        resnet = Profiles.resnet;
        resnet.data = resnet.data * Data_Scale;
        resnet.comp = resnet.comp * Comp_Scale;
        % Align T
        if length(resnet.data) > target_T
            resnet.data = resnet.data(1:target_T);
            resnet.comp = resnet.comp(1:target_T);
        elseif length(resnet.data) < target_T
             resnet.data = [resnet.data, zeros(1, target_T - length(resnet.data))];
             resnet.comp = [resnet.comp, zeros(1, target_T - length(resnet.comp))];
        end
    else
        % ResNet-50 Proxy (Deeper, less data than VGG)
        resnet.data = vgg16.data * 0.8; 
        resnet.comp = vgg16.comp * 1.2; 
    end
    
    if isfield(Profiles, 'mobilenet')
        mobilenet = Profiles.mobilenet;
        mobilenet.data = mobilenet.data * Data_Scale;
        mobilenet.comp = mobilenet.comp * Comp_Scale;
         % Align T
        if length(mobilenet.data) > target_T
            mobilenet.data = mobilenet.data(1:target_T);
            mobilenet.comp = mobilenet.comp(1:target_T);
        elseif length(mobilenet.data) < target_T
             mobilenet.data = [mobilenet.data, zeros(1, target_T - length(mobilenet.data))];
             mobilenet.comp = [mobilenet.comp, zeros(1, target_T - length(mobilenet.comp))];
        end
    else
        % MobileNet Proxy (Lightweight)
        mobilenet.data = vgg16.data * 0.2;
        mobilenet.comp = vgg16.comp * 0.1;
    end
    
    DNN_Data = {vgg16, resnet, mobilenet};
    
    % --- 2. Heterogeneous Fog Generation ---
    Fog = zeros(fog_num, 8);
    Fog(:, 1:2) = rand(fog_num, 2) * 1000;
    Fog(:, 3) = 500; % Radius
    
    % Node 1: high-capacity fog node
    Fog(1, 4) = 350.0;
    Fog(1, 7) = 350.0;
    Fog(1, 5) = 0.01;
    Fog(1, 6) = 40e6;
    Fog(1, 8) = 8192;
    
    % Node 2-3: compute-oriented nodes with narrower bandwidth
    for f = 2:3
        if f <= fog_num
            Fog(f, 4) = 250.0;
            Fog(f, 7) = 250.0;
            Fog(f, 5) = 0.1;
            Fog(f, 6) = 20e6;
            Fog(f, 8) = 4096;
        end
    end
    
    % Remaining nodes: communication-friendly but lower compute capacity
    for f = 4:fog_num
        Fog(f, 4) = 180.0;
        Fog(f, 7) = 180.0;
        Fog(f, 5) = 0.05;
        Fog(f, 6) = 40e6;
        Fog(f, 8) = 2048;
    end
    
    % --- 3. Mixed Task Generation ---
    Task = zeros(task_num, 5);
    % FIXED: ID in Col 1, Deadline in Col 3 (Standard Convention)
    Task(:, 1) = 1:task_num;
    
    % Distribution: 20% VGG16, 50% ResNet50, 30% MobileNetV2
    n_vgg = floor(task_num * 0.2);
    n_res = floor(task_num * 0.5);
    n_mob = task_num - n_vgg - n_res;
    
    idx = 1;
    % VGG16 (Type 1)
    Task(idx:idx+n_vgg-1, 2) = 1; 
    Task(idx:idx+n_vgg-1, 3) = 2.0;
    idx = idx + n_vgg;
    
    % ResNet50 (Type 2)
    Task(idx:idx+n_res-1, 2) = 2;
    Task(idx:idx+n_res-1, 3) = 1.0;
    idx = idx + n_res;
    
    % MobileNet (Type 3)
    Task(idx:end, 2) = 3;
    Task(idx:end, 3) = 0.5;
    
    % Shuffle Tasks to prevent order bias
    rand_perm = randperm(task_num);
    Task = Task(rand_perm, :);
    Task(:, 1) = 1:task_num; % Re-assign IDs sequentially
    
    % --- 4. Thing (User) Generation ---
    Thing = zeros(task_num, 6);
    Thing(:, 1:2) = rand(task_num, 2) * 1000;
    Thing(:, 4) = 5.0; % Local Speed (Pi 4B)
    Thing(:, 6) = 0.5; % Tx Power

    % --- 5. Build Pre-Calculated Matrices ---
    Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
end

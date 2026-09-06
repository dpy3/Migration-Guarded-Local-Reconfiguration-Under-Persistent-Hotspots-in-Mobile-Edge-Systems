function theta_out = DRL_DQN_Solver_Binary(Task, Fog, Thing, DNN_Data, max_episodes, Pre_In)
    % DRL_DQN_Solver_Binary.m
    % Binary Offloading (T=1) Version of Double DQN Solver
    %
    % Inputs:
    %   Task, Fog, Thing, DNN_Data: Standard inputs
    %   max_episodes: Number of training episodes (e.g., 200-500)
    %   Pre_In: Pre-computed structure (optional)
    %
    % Output:
    %   theta_out: N x M binary assignment matrix (Best found)
    
    [N, ~] = size(Task);
    [M, ~] = size(Fog);
    
    % --- Hyperparameters ---
    alpha = 0.001;       % Learning Rate
    gamma = 0.99;        % Discount Factor
    epsilon = 1.0;       % Exploration Rate
    epsilon_decay = 0.99;
    epsilon_min = 0.01;
    batch_size = 64;
    memory_capacity = 10000;
    target_update_freq = 10;
    
    % --- Pre-computation ---
    if nargin < 6 || isempty(Pre_In)
        Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
    else
        Pre = Pre_In;
    end
    
    % --- State & Action Space ---
    % State: [Data_i, Comp_i, Fog_Loads (M), Fog_Mems (M), Channel_Gains_i (M)]
    % Size: 2 + M + M + M = 2 + 3*M
    input_dim = 2 + 3 * M;
    hidden_dim = 128;
    output_dim = M; % 1..M=Fog (Mandatory Offloading to align with Baseline/ALM)
    
    % --- Initialization ---
    Q_Net = init_network(input_dim, hidden_dim, output_dim);
    Target_Net = Q_Net;
    Adam_State = init_adam(Q_Net);
    
    % Replay Memory
    Mem_State = zeros(memory_capacity, input_dim);
    Mem_Action = zeros(memory_capacity, 1);
    Mem_Reward = zeros(memory_capacity, 1);
    Mem_NextState = zeros(memory_capacity, input_dim);
    Mem_Done = zeros(memory_capacity, 1);
    mem_ptr = 1;
    mem_full = false;
    
    best_reward = -inf;
    best_theta = zeros(N, M);
    
    % --- Training Loop ---
    for episode = 1:max_episodes
        % Reset Environment for Episode
        % Current Loads on Fog Nodes (Accumulated Computation Time)
        current_fog_loads = zeros(1, M); 
        current_fog_mems = zeros(1, M);
        
        actions_episode = zeros(N, 1);
        
        for i = 1:N
            % 1. Construct State
            % Fix: Use Task(:, 4) for Data Size (MB) instead of Pre.Data
            data_size_mb = Task(i, 4);
            norm_data = data_size_mb / 200; % Normalize
            
            % Fix: Use Pre.Comp(i, :) average or specific node? 
            % Let's use average comp time across nodes as feature
            avg_comp = mean(Pre.Comp(i, :));
            norm_comp = avg_comp / 20;  
            
            norm_loads = current_fog_loads / 100; % Approximate normalization
            norm_mems = current_fog_mems / 4000;
            
            % Fix: Use Pre.Comm instead of Pre.B (Pre.B not available in Tier1)
            % Pre.Comm is Transmission Time.
            norm_comm = Pre.Comm(i, :) / 1.0; % Normalize (usually < 1s)
            
            state = [norm_data, norm_comp, norm_loads, norm_mems, norm_comm];
            
            % 2. Select Action (Epsilon-Greedy)
            if rand() < epsilon
                action = randi(output_dim);
            else
                q_values = forward_pass(Q_Net, state);
                [~, action] = max(q_values);
            end
            
            % 3. Execute Action
            act_idx = action; % 1..M (Direct Mapping)
            actions_episode(i) = act_idx;
            
            % Update Environment (Accumulate Load)
            % Fog Node act_idx
            f = act_idx;
            comp_time = Pre.Comp(i, f);
            current_fog_loads(f) = current_fog_loads(f) + comp_time;
            
            % Memory (Approximation)
            % Fix: Use Task(i, 4) for Data Size
            mem_usage = Task(i, 4) * 2.5; % 2.5x Data size
            current_fog_mems(f) = current_fog_mems(f) + mem_usage;
            
            % 4. Calculate Immediate Reward (Step-wise Latency Estimate)
            % Note: True latency depends on final load (M/D/1), but here we use current load approximation
            [step_reward, ~] = get_step_reward(act_idx, Pre, i, current_fog_loads, Task, Thing, Fog);
            
            % 5. Next State
            if i < N
                next_i = i + 1;
                n_data = Task(next_i, 4) / 200;
                n_comp = mean(Pre.Comp(next_i, :)) / 20;
                n_loads = current_fog_loads / 100;
                n_mems = current_fog_mems / 4000;
                n_comm = Pre.Comm(next_i, :) / 1.0;
                next_state = [n_data, n_comp, n_loads, n_mems, n_comm];
                done = 0;
            else
                next_state = zeros(1, input_dim);
                done = 1;
            end
            
            % 6. Store Experience
            idx = mem_ptr;
            Mem_State(idx, :) = state;
            Mem_Action(idx) = action;
            Mem_Reward(idx) = step_reward;
            Mem_NextState(idx, :) = next_state;
            Mem_Done(idx) = done;
            
            mem_ptr = mem_ptr + 1;
            if mem_ptr > memory_capacity
                mem_ptr = 1;
                mem_full = true;
            end
            
            % 7. Train
            curr_capacity = mem_full * memory_capacity + (~mem_full) * (mem_ptr - 1);
            if curr_capacity > batch_size
                batch_indices = randi(curr_capacity, batch_size, 1);
                b_states = Mem_State(batch_indices, :);
                b_actions = Mem_Action(batch_indices);
                b_rewards = Mem_Reward(batch_indices);
                b_next_states = Mem_NextState(batch_indices, :);
                b_dones = Mem_Done(batch_indices);
                
                % Double DQN Update
                Q_pred = forward_pass(Q_Net, b_states);
                Q_next_online = forward_pass(Q_Net, b_next_states);
                [~, max_actions_next] = max(Q_next_online, [], 2);
                Q_next_target = forward_pass(Target_Net, b_next_states);
                
                linear_idx = sub2ind(size(Q_next_target), (1:batch_size)', max_actions_next);
                q_target_values = Q_next_target(linear_idx);
                
                target_values = b_rewards + gamma * q_target_values .* (1 - b_dones);
                
                Q_target = Q_pred;
                for k = 1:batch_size
                    Q_target(k, b_actions(k)) = target_values(k);
                end
                
                grads = backward_pass(Q_Net, b_states, Q_target);
                [Q_Net, Adam_State] = adam_update(Q_Net, grads, Adam_State, alpha);
            end
        end
        
        % Update Target Net
        if mod(episode, target_update_freq) == 0
            Target_Net = Q_Net;
        end
        
        % Decay Epsilon
        if epsilon > epsilon_min
            epsilon = epsilon * epsilon_decay;
        end
        
        % Evaluate Episode
        theta_ep = actions_to_theta_binary(actions_episode, N, M);
        
        % Use Rigorous Metric for Best Model Selection
        % Note: calculate_metrics_v2 returns [avg_satis, viol_rate, ...]
        % We want to maximize Satisfaction - Penalty * Violation
        [avg_sat, viol, ~, ~, ~, ~] = calculate_metrics_v2(theta_ep, Task, Fog, Thing, DNN_Data, Pre);
        
        % Objective: Maximize Sat, Minimize Viol
        episode_score = avg_sat - viol * 10; % Simple scalar score
        
        if episode_score > best_reward
            best_reward = episode_score;
            best_theta = theta_ep;
        end
    end
    
    theta_out = best_theta;
end

% --- Helper Functions ---

function [reward, cost] = get_step_reward(act_idx, Pre, i, current_loads, Task, Thing, Fog)
    % Estimate Latency for Step Reward
    % Mandatory Offloading (Fog 1..M)
    
    f = act_idx;
    t_comm = Pre.Comm(i, f);
    t_comp = Pre.Comp(i, f);
    
    % Wait Time Approximation (based on current load seen SO FAR)
    load_time = current_loads(f);
    t_wait = load_time; % FIFO approximation
    
    cost = t_comm + t_comp + t_wait;
    
    deadline = Task(i, 3);
    penalty = 0;
    if cost > deadline
        penalty = 10;
    end
    
    reward = -(cost + penalty);
end

function theta = actions_to_theta_binary(actions, N, M)
    theta = zeros(N, M);
    for i = 1:N
        if actions(i) > 0
            theta(i, actions(i)) = 1;
        end
    end
end

function Net = init_network(in, hid, out)
    Net.W1 = randn(in, hid) * sqrt(2/in);
    Net.b1 = zeros(1, hid);
    Net.W2 = randn(hid, out) * sqrt(2/hid);
    Net.b2 = zeros(1, out);
end

function Adam = init_adam(Net)
    Adam.mW1 = zeros(size(Net.W1)); Adam.vW1 = zeros(size(Net.W1));
    Adam.mb1 = zeros(size(Net.b1)); Adam.vb1 = zeros(size(Net.b1));
    Adam.mW2 = zeros(size(Net.W2)); Adam.vW2 = zeros(size(Net.W2));
    Adam.mb2 = zeros(size(Net.b2)); Adam.vb2 = zeros(size(Net.b2));
    Adam.t = 0;
end

function q_values = forward_pass(Net, X)
    Z1 = X * Net.W1 + Net.b1;
    A1 = max(0, Z1);
    Z2 = A1 * Net.W2 + Net.b2;
    q_values = Z2;
end

function grads = backward_pass(Net, X, Y_target)
    batch_size = size(X, 1);
    Z1 = X * Net.W1 + Net.b1;
    A1 = max(0, Z1);
    Y_pred = A1 * Net.W2 + Net.b2;
    
    dY = 2 * (Y_pred - Y_target) / batch_size;
    
    grads.dW2 = A1' * dY;
    grads.db2 = sum(dY, 1);
    
    dA1 = dY * Net.W2';
    dZ1 = dA1 .* (Z1 > 0);
    
    grads.dW1 = X' * dZ1;
    grads.db1 = sum(dZ1, 1);
end

function [Net, Adam] = adam_update(Net, grads, Adam, alpha)
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
    Adam.t = Adam.t + 1;
    
    [Net.W1, Adam.mW1, Adam.vW1] = adam_step(Net.W1, grads.dW1, Adam.mW1, Adam.vW1, Adam.t, alpha, beta1, beta2, eps);
    [Net.b1, Adam.mb1, Adam.vb1] = adam_step(Net.b1, grads.db1, Adam.mb1, Adam.vb1, Adam.t, alpha, beta1, beta2, eps);
    [Net.W2, Adam.mW2, Adam.vW2] = adam_step(Net.W2, grads.dW2, Adam.mW2, Adam.vW2, Adam.t, alpha, beta1, beta2, eps);
    [Net.b2, Adam.mb2, Adam.vb2] = adam_step(Net.b2, grads.db2, Adam.mb2, Adam.vb2, Adam.t, alpha, beta1, beta2, eps);
end

function [param, m, v] = adam_step(param, grad, m, v, t, alpha, b1, b2, eps)
    m = b1 * m + (1 - b1) * grad;
    v = b2 * v + (1 - b2) * grad.^2;
    m_hat = m / (1 - b1^t);
    v_hat = v / (1 - b2^t);
    param = param - alpha * m_hat ./ (sqrt(v_hat) + eps);
end

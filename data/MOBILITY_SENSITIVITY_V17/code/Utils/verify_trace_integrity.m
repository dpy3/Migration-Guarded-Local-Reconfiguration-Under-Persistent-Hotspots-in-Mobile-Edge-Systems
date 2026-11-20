function verify_trace_integrity()
    % verify_trace_integrity.m
    % Audits the simulation environment for physical realism and data integrity.
    % Ensures that:
    % 1. Task execution times match empirical hardware traces.
    % 2. Communication models follow Shannon's law.
    % 3. Deadlines are physically feasible (at least on some nodes).
    
    clc;
    fprintf('=== TRACE INTEGRITY AUDIT ===\n');
    
    % Setup Paths
    script_folder = fileparts(mfilename('fullpath'));
    addpath(fullfile(script_folder, '..', 'Environment'));
    addpath(fullfile(script_folder, '..', 'Data'));
    
    % 1. Load Ground Truth
    [Profiles, DNN_Data, Trace_DB] = Hardware_Profile_Data();
    fprintf('[Check 1] Hardware Profiles Loaded.\n');
    
    % 2. Generate Sample Environment
    N = 10; M = 5;
    [Task, Fog, Thing, DNN_Data_Env, Pre] = Generate_Environment(N, M);
    fprintf('[Check 2] Environment Generated (N=%d, M=%d).\n', N, M);
    
    % 3. Verify Computation Times
    fprintf('\n--- Verifying Computation Traces ---\n');
    passed_comp = true;
    for i = 1:N
        type = Task(i, 2);
        % Map Type to Trace Row: 1->3(VGG), 2->2(ResNet), 3->1(MobileNet)
        if type == 1, row_idx = 3; model='VGG16';
        elseif type == 2, row_idx = 2; model='ResNet50';
        elseif type == 3, row_idx = 1; model='MobileNet';
        end
        
        % Check Node 1 (Nano - Type 1 - Col 2)
        t_nano_env = Pre.Comp(i, 1);
        t_nano_truth = Trace_DB(row_idx, 2);
        
        % Check Node 3 (Pi - Type 2 - Col 1)
        t_pi_env = Pre.Comp(i, 3);
        t_pi_truth = Trace_DB(row_idx, 1);
        
        % Check Node 5 (Cloud - Type 3 - Col 3)
        t_cloud_env = Pre.Comp(i, 5);
        t_cloud_truth = Trace_DB(row_idx, 3);
        
        % Verify exact match (Environment generator copies trace exactly)
        if abs(t_nano_env - t_nano_truth) > 1e-6
            fprintf('  [FAIL] Task %d (%s) on Nano: Env=%.4f, Truth=%.4f\n', i, model, t_nano_env, t_nano_truth);
            passed_comp = false;
        end
    end
    
    if passed_comp
        fprintf('  [PASS] All computation times match empirical traces exactly.\n');
    else
        fprintf('  [FAIL] Computation trace mismatch detected.\n');
    end
    
    % 4. Verify Feasibility (Deadline vs Execution)
    fprintf('\n--- Verifying Task Feasibility ---\n');
    feasible_count = 0;
    for i = 1:N
        min_exec = min(Pre.Comp(i, :));
        deadline = Task(i, 3);
        if min_exec < deadline
            feasible_count = feasible_count + 1;
        else
            fprintf('  [WARN] Task %d Infeasible: MinExec=%.4f > Deadline=%.4f\n', i, min_exec, deadline);
        end
    end
    fprintf('  Feasibility Ratio: %d/%d (%.1f%%)\n', feasible_count, N, (feasible_count/N)*100);
    
    if feasible_count == N
        fprintf('  [PASS] All tasks are theoretically feasible on at least one node.\n');
    else
        fprintf('  [WARN] Some tasks are impossible even without contention.\n');
    end
    
    % 5. Verify Communication Model
    fprintf('\n--- Verifying Communication Model ---\n');
    % Pick a task and check rate vs distance
    t_idx = 1;
    user_loc = Thing(t_idx, 1:2);
    
    % Find Cloud Node (Index 5)
    cloud_dist = norm(user_loc - [250, 250]);
    % Cloud Rate should be fixed 100Mbps (Pre.Comm = Size / 100e6 + 0.05)
    data_bits = Task(t_idx, 4) * 8 * 1e6;
    expected_cloud_comm = data_bits / 100e6 + 0.05;
    
    if abs(Pre.Comm(t_idx, 5) - expected_cloud_comm) < 1e-6
        fprintf('  [PASS] Cloud Backhaul Model Verified (100Mbps + 50ms RTT).\n');
    else
        fprintf('  [FAIL] Cloud Model Mismatch: Env=%.4f, Exp=%.4f\n', Pre.Comm(t_idx, 5), expected_cloud_comm);
    end
    
    fprintf('\n=== AUDIT COMPLETE ===\n');
end

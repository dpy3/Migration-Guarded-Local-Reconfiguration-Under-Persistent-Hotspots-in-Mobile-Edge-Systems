function [theta, info] = BigMEC_Greedy_Adapter_Binary(Task, Fog, Thing, DNN_Data, Pre, theta_previous)
% Equal-priority, non-displacing adapter of the public BigMEC greedy baseline.
% Source contract: highest utility in a destination neighborhood, subject to
% available memory, when a user's attachment changes. In this simulator every
% task receives a mobility update each slot, all five nodes form the candidate
% neighborhood, and utility is negative communication-plus-compute latency.
% Paper: Brandherm et al., IEEE/ACM SEC 2022, DOI 10.1109/SEC54971.2022.00018.
% Public code: https://github.com/flbrandh/MEC-Simulator-2-BigMEC

[N, M] = size(Pre.Comp);
if nargin < 6 || isempty(theta_previous) || ~isequal(size(theta_previous), [N, M])
    theta = MPC_Rollout_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre, struct());
    theta = Perform_Safe_Harbor_Repair(theta, Pre, Fog, Task, DNN_Data);
else
    theta = theta_previous;
end

if size(Fog, 2) >= 8, mem_caps = Fog(:, 8)'; else, mem_caps = Fog(:, end)'; end
task_mem = zeros(N, 1);
for i = 1:N, task_mem(i) = local_task_memory(Task(i, 2), DNN_Data); end
node_mem = zeros(1, M);
for i = 1:N
    current = find(theta(i, :) > 0, 1);
    if ~isempty(current), node_mem(current) = node_mem(current) + task_mem(i); end
end

evaluations = 0;
moves = 0;
for i = 1:N
    current = find(theta(i, :) > 0, 1);
    if isempty(current), continue; end
    best = current;
    best_cost = Pre.Comm(i, current) + Pre.Comp(i, current);
    for f = 1:M
        evaluations = evaluations + 1;
        projected_mem = node_mem(f) + task_mem(i) - double(f == current) * task_mem(i);
        if projected_mem > mem_caps(f), continue; end
        cost = Pre.Comm(i, f) + Pre.Comp(i, f);
        if cost < best_cost - 1e-12
            best = f;
            best_cost = cost;
        end
    end
    if best ~= current
        theta(i, :) = 0;
        theta(i, best) = 1;
        node_mem(current) = node_mem(current) - task_mem(i);
        node_mem(best) = node_mem(best) + task_mem(i);
        moves = moves + 1;
    end
end
info = struct('CandidateEvaluations', evaluations, 'ProposedMoves', moves, ...
    'AdapterName', "BigMECGreedy-EqualPriority-NoDisplacement");
end

function value = local_task_memory(task_type, DNN_Data)
if iscell(DNN_Data) && task_type <= numel(DNN_Data) && isfield(DNN_Data{task_type}, 'data')
    value = sum(DNN_Data{task_type}.data);
elseif isnumeric(DNN_Data)
    map = [3, 2, 1]; row = map(min(max(round(task_type), 1), 3));
    value = sum(DNN_Data(row, 1:min(2, size(DNN_Data, 2))));
else
    value = 100;
end
end

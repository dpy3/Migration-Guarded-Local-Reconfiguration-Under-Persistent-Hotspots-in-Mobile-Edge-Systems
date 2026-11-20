function theta_out = LP_Relaxed_Solver_Binary(Task, Fog, Thing, DNN_Data, Pre_In)
% LP relaxation reference: linear load/memory proxy with admission variables.
[N, ~] = size(Task); M = size(Fog, 1);
if nargin < 5 || isempty(Pre_In)
    Pre = build_pre_struct_tier1(Task, Fog, Thing, DNN_Data);
else
    Pre = Pre_In;
end
assert(exist('linprog', 'file') == 2, 'MCA:MissingLinprog', ...
    'Optimization Toolbox (linprog) is required for LPRelax.');

mean_deadline = max(mean(Task(:, 3)), 1e-6);
cost = Pre.Comm + Pre.Comp;
% y_i denotes whether task i is admitted by the relaxation. The binary
% restriction on x is relaxed, then the x block is deterministically rounded.
f = [cost(:); -1000 * ones(N, 1)];
A = zeros(2 * M, N * M + N); b = zeros(2 * M, 1);
mem_caps = get_mem_caps(Fog, M);
task_mem = zeros(N, 1);
for i = 1:N, task_mem(i) = get_task_mem(Task(i, 2), DNN_Data); end
for node = 1:M
    cols = node:M:N * M;
    A(node, cols) = Pre.Comp(:, node)';
    b(node) = 0.97 * mean_deadline;
    A(M + node, cols) = task_mem';
    b(M + node) = mem_caps(node);
end
Aeq = [kron(ones(1, M), eye(N)), -eye(N)]; beq = zeros(N, 1);
opts = optimoptions('linprog', 'Display', 'none');
[sol, ~, exitflag] = linprog(f, A, b, Aeq, beq, ...
    zeros(N * M + N, 1), ones(N * M + N, 1), opts);
if exitflag <= 0
    error('MCA:LPFailure', 'LP relaxation failed with exitflag=%d.', exitflag);
end
x = reshape(sol(1:N * M), N, M);
theta_out = zeros(N, M);
for i = 1:N
    [~, order] = sortrows([-x(i, :)' cost(i, :)'], [1 2], {'ascend', 'ascend'});
    theta_out(i, order(1)) = 1;
end
end

function caps = get_mem_caps(Fog, M)
if size(Fog, 2) >= 8, caps = Fog(:, 8)'; else, caps = Fog(:, end)'; end
caps = reshape(caps, 1, []);
if numel(caps) < M, caps(end + 1:M) = inf; end
caps = caps(1:M);
end

function mem_mb = get_task_mem(t_type, DNN_Data)
row_idx = max(1, min(3, 4 - t_type));
if isnumeric(DNN_Data) && size(DNN_Data, 1) >= row_idx && size(DNN_Data, 2) >= 2
    mem_mb = DNN_Data(row_idx, 1) + DNN_Data(row_idx, 2);
else
    mem_mb = 100;
end
end

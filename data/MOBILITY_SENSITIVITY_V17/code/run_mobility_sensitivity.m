function run_mobility_sensitivity(mode)
% Run paired mobility-intensity smoke or formal sensitivity on a frozen copy.
if nargin < 1 || isempty(mode), mode = 'mobility_smoke'; end
mode = lower(char(mode));
assert(ismember(mode, {'mobility_smoke','mobility_formal'}), ...
    'Mobility:InvalidMode', 'Mode must be mobility_smoke or mobility_formal.');

root = fileparts(mfilename('fullpath'));
logfile = fullfile(root, [upper(mode) '_CONSOLE.log']);
diary(logfile);
diary on;
cleanup = onCleanup(@() diary('off')); %#ok<NASGU>
sigmas_m = [15, 35, 70];
scenarios = {'compute_hotspot','link_hotspot','coupled_hotspot'};
fprintf('Mobility sensitivity start: mode=%s, sigmas_m=%s, time=%s\n', ...
    mode, mat2str(sigmas_m), datestr(now, 31));
for i = 1:numel(sigmas_m)
    fprintf('\n=== sigma=%g m (%d/%d) ===\n', sigmas_m(i), i, numel(sigmas_m));
    Run_Paper2_Revision(mode, [], scenarios, [], 0.05, sigmas_m(i));
end
fprintf('\nMobility sensitivity finished: mode=%s, time=%s\n', mode, datestr(now, 31));
end

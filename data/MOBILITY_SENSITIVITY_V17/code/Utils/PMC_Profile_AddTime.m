function PMC_Profile_AddTime(field_name, timer_handle)
% Accumulate profiling time in the current experiment counter struct.
global PMC_AUDIT_COUNTERS;
if isempty(PMC_AUDIT_COUNTERS), PMC_AUDIT_COUNTERS = struct(); end
if ~isfield(PMC_AUDIT_COUNTERS, field_name), PMC_AUDIT_COUNTERS.(field_name) = 0; end
PMC_AUDIT_COUNTERS.(field_name) = PMC_AUDIT_COUNTERS.(field_name) + toc(timer_handle);
end

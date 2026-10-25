function tbl = diagnose_halfspace_dd_rank_limit(tag, names)
%DIAGNOSE_HALFSPACE_DD_RANK_LIMIT Exact allocation on fixed saved trajectories.
% This is a diagnostic lower bound, not a retrained alternative DD solver.
    if nargin < 1, tag = 'pilot'; end
    if nargin < 2
        names = {'material3-greedy','slabs2-aligned-sym8','slabs3-aligned-sym8'};
    end
    out = helm_output_path('outputs','halfspace_dd_refinement',char(tag));
    rows = cell(numel(names)*3,1);
    pos = 0;
    for name = string(names)
        saved = load(fullfile(out,char(name)+".mat"),'artifact');
        a = saved.artifact;
        for j = 1:numel(a.layers)
            layer = a.layers{j};
            old_space = layer.basis_space;
            cfg = struct('rank_global_tail_tol',a.cfg.corr_block3_global_tail_tols(j), ...
                'rank_global_tail_strategy','minimum_storage');
            t = tic;
            optimal = rom_reduced_space_utils('build_block3_space', ...
                layer.eta_snapshots,old_space.partition,cfg);
            old_entries = sum(layer.diagnostics.ranks.*layer.diagnostics.dofs);
            if strcmp(optimal.type,'block3')
                new_entries = numel(optimal.Un)+numel(optimal.Uf)+numel(optimal.Up);
            else
                new_entries = sum([optimal.blocks.k].*[optimal.blocks.count]);
            end
            assert(new_entries <= old_entries);
            pos = pos+1;
            rows{pos} = struct('variant',name,'layer',j,'current_entries',old_entries, ...
                'optimal_entries',new_entries, 'extra_saving_percent', ...
                100*(1-new_entries/old_entries),'diagnostic_s',toc(t));
            fprintf('%s layer %d: current=%d, exact=%d\n',name,j,old_entries,new_entries);
        end
    end
    tbl = struct2table(vertcat(rows{1:pos}));
    writetable(tbl,fullfile(out,'fixed_trajectory_rank_limit.csv'));
    save(fullfile(out,'fixed_trajectory_rank_limit.mat'),'tbl','names');
end

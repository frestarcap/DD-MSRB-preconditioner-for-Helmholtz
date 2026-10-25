function export_ecc_rank_allocation_inputs
% Export frozen ECC local spectra for a same-snapshot greedy/DP audit.
% No training, online solves, or modifications of saved models are performed.
    source_dir=helm_output_path('outputs','ecc_dd_budget_20260919');
    out=helm_output_path('outputs','greedy_dp_audit_20260920');
    if ~isfolder(out),mkdir(out);end
    tags={'uniform5','uniform6','last4','last3','layered','uniform3','two6'};
    records={};
    for ti=1:numel(tags)
        source=fullfile(source_dir,[tags{ti} '_dd.mat']);
        s=load(source,'model','cfg');
        for j=1:numel(s.model.layers)
            sp=s.model.layers{j}.space;
            assert(strcmp(sp.type,'block3'));
            assert(strcmp(s.cfg.corr_block3_global_tail_strategy,'marginal_storage'));
            for key={'near','far','pml'}
                assert(isempty(s.cfg.(['corr_block3_tail_tol_' key{1}])));
                assert(isinf(s.cfg.(['corr_block3_rank_cap_' key{1}])));
            end
            rec=struct('tag',tags{ti},'layer',j,'source',source, ...
                'eta',s.cfg.corr_rank_rtols(j),'eta_hex',num2hex(s.cfg.corr_rank_rtols(j)), ...
                'n',[numel(sp.partition.idx_near),numel(sp.partition.idx_far),numel(sp.partition.idx_pml)], ...
                'saved_ranks',[sp.kn,sp.kf,sp.kp], ...
                'svals',{{sp.block_meta.near.singular_values(:).', ...
                          sp.block_meta.far.singular_values(:).', ...
                          sp.block_meta.pml.singular_values(:).'}});
            rec.svals_hex=cellfun(@(v)cellstr(num2hex(v(:))),rec.svals,'UniformOutput',false);
            records{end+1}=rec; %#ok<AGROW>
        end
    end
    payload=struct('matlab',version,'scope','Frozen layer snapshots; basis entries only', ...
        'records',{records});
    fid=fopen(fullfile(out,'inputs.json'),'w');
    assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(payload));
    fprintf('ECC_RANK_INPUT_EXPORT_PASS layers=%d\n',numel(records));
end

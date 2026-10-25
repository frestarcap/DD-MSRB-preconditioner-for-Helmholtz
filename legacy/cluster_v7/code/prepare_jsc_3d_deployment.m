function files=prepare_jsc_3d_deployment(candidate,problem_name)
%PREPARE_JSC_3D_DEPLOYMENT Export one lean frozen model per process benchmark.
    if nargin<2, problem_name='lens'; end
    out=helm_output_path('outputs','revision_20260907','deployment');
    if ~exist(out,'dir'), mkdir(out); end
    if strcmp(problem_name,'lens')
        source=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
            ['gaussian_lens_jsc_20260907_lens_' candidate '_training.mat']);
    else
        source=candidate;
    end
    saved=load(source,'models','cfg','resolved_problem_cfg','rhs');
    cfg=saved.cfg; problem_cfg=saved.resolved_problem_cfg; rhs=saved.rhs;
    filelabel=candidate;
    if ~strcmp(problem_name,'lens'), [~,filelabel]=fileparts(candidate); end
    files=strings(0);
    for key=fieldnames(saved.models).'
        name=key{1}; model=saved.models.(name);
        original_model_summary=rmfield(model,'layers');
        layers=cell(size(model.layers));
        for j=1:numel(layers)
            l=model.layers{j}; sp=l.space;
            if strcmp(sp.type,'dense')
                sp=rom_reduced_space_utils('make_dense_space',sp.basis);
            else
                sp=rmfield(sp,intersect(fieldnames(sp),{'partition','meta','block_meta'}));
                for b=1:numel(sp.blocks)
                    sp.blocks(b).meta=[];
                end
            end
            layers{j}=struct('space',sp,'Ar0',l.Ar0,'Ar1',l.Ar1,'rank',l.rank);
        end
        model=struct('layers',{layers},'num_layers',numel(layers));
        file=fullfile(out,[problem_name '_' filelabel '_' name '.mat']);
        save(file,'model','cfg','problem_cfg','rhs','source','original_model_summary','-v7.3');
        files(end+1)=string(file); %#ok<AGROW>
    end
end

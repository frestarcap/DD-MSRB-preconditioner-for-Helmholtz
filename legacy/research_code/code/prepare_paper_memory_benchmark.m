function prepare_paper_memory_benchmark
% Export lean model-only files before launching measured fresh processes.
    out=helm_output_path('outputs','paper_memory_benchmark');
    if ~exist(out,'dir'), mkdir(out); end
    p=load(helm_output_path('outputs','halfspace_dd_paper','paper_data.mat'),'paper');
    for m=1:2
        names={'global','dd'}; name=names{m};
        s=load(p.paper.source_files{m},'artifact');
        cfg=s.artifact.cfg; layers=s.artifact.layers;
        save_pair('halfspace',name,cfg,layers,'basis_space',out);
    end
    s=load(helm_output_path('outputs','ecc_two_layer_paper_20260911','paper_data.mat'), ...
        'global_results','dd_results');
    names={'global','dd'};
    for m=1:2
        r=s.([names{m} '_results']);
        save_pair('ecc',names{m},r.cfg,r.correction_rom.layers,'basis_space',out);
    end
    source=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
        'gaussian_waveguide_waveguide_dd_memory_saving_41pct_reproduction_training.mat');
    files=prepare_jsc_3d_deployment(source,'waveguide');
    for m=1:numel(files)
        [~,stem]=fileparts(files(m));
        if endsWith(stem,'_global'), name='global'; else, name='dd'; end
        copyfile(files(m),fullfile(out,['waveguide_' name '.mat']));
    end
end
function save_pair(problem,name,cfg,layers,field,out)
    for j=1:numel(layers)
        l=layers{j}; sp=l.(field);
        sp=rmfield(sp,intersect(fieldnames(sp),{'meta','block_meta'}));
        if strcmp(sp.type,'blockn')
            sp=rmfield(sp,intersect(fieldnames(sp),{'partition'}));
            sp.blocks=rmfield(sp.blocks,intersect(fieldnames(sp.blocks),{'meta'}));
        elseif strcmp(sp.type,'block3')
            keep={'idx_near','idx_far','idx_pml','count_near','count_far','count_pml'};
            sp.partition=rmfield(sp.partition,setdiff(fieldnames(sp.partition),keep));
        end
        l=struct('basis_space',sp);
        if isfield(layers{j},'Ar_base')
            l.Ar_base=layers{j}.Ar_base; l.Ar_right=layers{j}.Ar_right;
        end
        layers{j}=l;
    end
    model=struct('layers',{layers},'num_layers',numel(layers));
    cfg.parallel_test=false; cfg.parallel_train=false;
    cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
    cfg.record_true_helm_history=false;
    save(fullfile(out,[problem '_' name '.mat']),'cfg','model','-v7.3');
end

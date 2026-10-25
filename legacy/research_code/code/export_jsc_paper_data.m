function export_jsc_paper_data
%EXPORT_JSC_PAPER_DATA Freeze the chosen PC model and measured phase timings.
    out=helm_output_path('outputs','revision_20260907','pc');
    names={'global','bands1_axial1_signed0'}; pair=cell(1,2); rows=table(); layers=table();
    for m=1:2
        source=fullfile(out,[names{m} '_verified.mat']); d=load(source,'r'); r=d.r;
        timing=load(fullfile(out,[names{m} '_timings.mat']));
        assert(size(timing.samples,1)==5);
        r.test.online_total_time=mean(timing.samples,1).';
        r.test.online_precorr_time=mean(timing.reduced_samples,1).';
        r.test.online_lift_time=mean(timing.lift_samples,1).';
        r.test.online_fp_time=mean(timing.completion_samples,1).';
        storage=rom_deployment_storage(r.correction_rom.layers);
        rho=zeros(numel(r.test.params),1);
        for k=1:numel(rho)
            A=r.problem.A_base+(1/r.test.params(k)^2-1)*r.problem.A_right;
            rho(k)=norm(r.rhs-A*r.test.online_final_u(:,k))/norm(r.rhs);
        end
        assert(all(rho<=1e-6));
        row=table(string(names{m}),storage.basis_MiB,storage.operators_MiB, ...
            storage.index_MiB,storage.payload_MiB,r.timing.offline, ...
            mean(r.test.online_precorr_time),mean(r.test.online_lift_time), ...
            mean(r.test.online_fp_time),mean(r.test.online_total_time), ...
            max(std(timing.samples,0,1)),mean(r.test.online_fp_iter), ...
            max(r.test.online_fp_iter),mean(rho),max(rho), ...
            mean(r.test.online_precorr_relres_final), ...
            mean(r.test.online_final_err_helm_full),max(r.test.online_final_err_helm_full), ...
            'VariableNames',{'method','basis_MiB','operators_MiB','index_MiB', ...
            'payload_MiB','offline_s','reduced_s','lift_s','completion_s','total_s', ...
            'max_query_sd','completion_mean','completion_max','mean_true_residual', ...
            'max_true_residual','mean_precompletion_residual','mean_error','max_error'});
        rows=[rows;row]; %#ok<AGROW>
        for j=1:numel(r.correction_rom.layers)
            sp=r.correction_rom.layers{j}.basis_space;
            if strcmp(sp.type,'dense')
                count=numel(sp.basis); ranks=sp.k; tail=NaN;
            else
                count=sum(arrayfun(@(b)numel(b.basis),sp.blocks));
                ranks=[sp.blocks.k]; tail=sp.blocks(1).meta.global_relative_tail;
            end
            layers=[layers;table(string(names{m}),j,string(mat2str(ranks)),count,tail, ...
                'VariableNames',{'method','layer','ranks','entries','tail'})]; %#ok<AGROW>
        end
        r.revision_provenance=struct('model_file',source, ...
            'timing_file',fullfile(out,[names{m} '_timings.mat']), ...
            'true_acceptance',1e-6,'native_initial_tolerance',1e-6, ...
            'repetitions',5,'timing_threads',1);
        pair{m}=r;
    end
    global_results=pair{1}; dd_results=pair{2}; %#ok<NASGU>
    base_cfg=global_results.cfg; %#ok<NASGU>
    writetable(rows,fullfile(out,'paper_summary.csv')); writetable(layers,fullfile(out,'paper_layers.csv'));
    save(fullfile(out,'paper_data.mat'),'global_results','dd_results','base_cfg','-v7.3');
    sp=dd_results.correction_rom.layers{1}.basis_space;
    ids=zeros(dd_results.problem.Ntot,1);
    for j=1:numel(sp.blocks), ids(sp.blocks(j).idx)=j; end
    fig=figure('Color','w','Position',[100,100,1150,850],'Visible','off');
    ax=axes(fig); imagesc(ax,dd_results.problem.x_int,dd_results.problem.y_int, ...
        reshape(ids,dd_results.problem.nx,dd_results.problem.ny).');
    axis(ax,'image'); set(ax,'YDir','normal','FontSize',22,'LineWidth',1.2);
    colormap(ax,[0.84 .36 .32;.4 .65 .85;.85 .75 .43;.93 .89 .72;.42 .7 .5;.76 .87 .78]);
    clim(ax,[.5,6.5]); cb=colorbar(ax); cb.Ticks=1:6;
    cb.TickLabels={'Core','Cladding','LR: inner','LR: outer','TB: inner','TB: outer'};
    cb.FontSize=18; xlabel(ax,'x'); ylabel(ax,'y'); title(ax,'PC: six-block depth partition');
    target=helm_output_path('LaTeX_DL_468198_240419','figures', ...
        'photonic_crystal_correction_rom_4block','pc_four_block_dd_partition.png');
    exportgraphics(fig,target,'Resolution',300); close(fig);
end

function summary=run_jsc_pc_study(mode)
%RUN_JSC_PC_STUDY Geometry screening followed by independent trajectory replay.
    if nargin<1, mode='screen'; end
    out=helm_output_path('outputs','revision_20260907','pc');
    if ~exist(out,'dir'), mkdir(out); end
    source=helm_output_path('outputs','helm_trajectory_evidence', ...
        'review_repaired_20260905','pc_residual_ablation_paper.mat');
    old=load(source,'global_results','dd_results'); g=old.global_results; d=old.dd_results;
    variants=struct('name',{},'cfg',{});
    for nb=[1,3]
        for ax=[1,3]
            for signed=[false,true]
                cfg=g.cfg; cfg.block3_partition_mode='pc_refined';
                cfg.pc_dd_band_edges=0.75;
                if nb==3, cfg.pc_dd_band_edges=[0.75,1.75,2.75]; end
                cfg.pc_dd_axial_sections=ax; cfg.pc_dd_depth_edges=0.5;
                cfg.pc_dd_signed_pml=signed;
                name=sprintf('bands%d_axial%d_signed%d',nb,ax,signed);
                variants(end+1)=struct('name',name,'cfg',cfg); %#ok<AGROW>
            end
        end
    end
    if strcmp(mode,'screen')
        summary=table();
        for v=1:numel(variants)
            part=build_pc_refined_partition(g.problem,variants(v).cfg);
            for traj=1:2
                r=g; if traj==2, r=d; end
                ls=r.correction_rom.layers;
                entries=0; reduced=0; ranks=[]; block_detail=table();
                for j=1:3
                    opts=struct('rank_global_tail_tol',g.cfg.corr_pod_singular_value_tols(j), ...
                        'rank_global_tail_strategy','marginal_storage');
                    sp=rom_reduced_space_utils('build_blockn_space',ls{j}.eta_snapshots,part,opts);
                    entries=entries+sum(arrayfun(@(b)numel(b.basis),sp.blocks));
                    ranks(j)=sum([sp.blocks.k]); reduced=reduced+2*ranks(j)^2;
                    for b=1:numel(sp.blocks)
                        B=sp.blocks(b); energy=norm(ls{j}.eta_snapshots(B.idx,:),'fro')^2;
                        block_detail=[block_detail;table(j,string(B.name),B.count,B.k,energy, ...
                            'VariableNames',{'layer','block','dofs','rank','snapshot_energy'})]; %#ok<AGROW>
                    end
                end
                s=rom_deployment_storage(g.correction_rom.layers);
                payload=(16*(entries+reduced)+8*part.n)/2^20;
                summary=[summary;table(string(variants(v).name),traj,entries,payload, ...
                    string(mat2str(ranks)),100*(1-16*entries/2^20/s.basis_MiB), ...
                    'VariableNames',{'partition','snapshot_trajectory','basis_entries', ...
                    'payload_MiB','total_local_ranks','saving_vs_independent_global_percent'})]; %#ok<AGROW>
                writetable(block_detail,fullfile(out,sprintf('%s_traj%d_blocks.csv',variants(v).name,traj)));
            end
        end
        writetable(summary,fullfile(out,'screen.csv')); save(fullfile(out,'screen.mat'),'summary','variants');
        disp(summary); return;
    end
    screened=load(fullfile(out,'screen.mat'),'summary','variants');
    tableg=screened.summary(screened.summary.snapshot_trajectory==1,:);
    tableg=sortrows(tableg,'payload_MiB'); top=tableg.partition(1:3);
    selected=find(ismember(string({variants.name}), ...
        [top;"bands1_axial1_signed0";"bands1_axial1_signed1";"bands3_axial1_signed0"]));
    base=g.cfg; base.parallel_train=false; base.parallel_test=false;
    base.run_wave_baselines=false; base.run_nr_rom_comparison=false;
    base.do_plots=false; base.save_plots=false; base.save_benchmark=false;
    base.wh_tol=1e-7; base.online_helm_accept_tol=1e-6;
    base.reference_mode='helm_lift'; base.offline_reference_solver='helmholtz';
    base.use_direct_rom_init=false; base.build_direct_rom_when_unused=false;
    candidates=struct('name','global','cfg',base);
    c=base; c.corr_basis_style='block3';
    c.block3_partition_mode='pc_core_cladding_lr_tb_pml';
    candidates(end+1)=struct('name','current4','cfg',c);
    for v=selected
        c=base; c.corr_basis_style='block3';
        for field={'block3_partition_mode','pc_dd_band_edges','pc_dd_axial_sections', ...
                'pc_dd_depth_edges','pc_dd_signed_pml'}
            c.(field{1})=variants(v).cfg.(field{1});
        end
        candidates(end+1)=struct('name',variants(v).name,'cfg',c); %#ok<AGROW>
    end
    summary=table();
    for v=1:numel(candidates)
        file=fullfile(out,[candidates(v).name '.mat']);
        if isfile(file)
            saved=load(file,'r'); r=saved.r;
        else
            r=helm_waveholtz_correction_rom(candidates(v).cfg); save(file,'r','-v7.3');
        end
        if ismember(mode,{'verified','timing'})
            verified_file=fullfile(out,[candidates(v).name '_verified.mat']);
            if isfile(verified_file)
                saved=load(verified_file,'r'); r=saved.r;
            else
                offline=r.timing.offline; cfg=r.cfg;
                cfg.wh_tol=1e-6; cfg.online_true_helm_tol=1e-6;
                cfg.online_helm_accept_tol=1e-6;
                r=helm_waveholtz_correction_rom(cfg,r); r.timing.offline=offline;
                save(verified_file,'r','-v7.3');
            end
        end
        times=r.test.online_total_time; reps=1; maxsd=NaN;
        if strcmp(mode,'timing') && ismember(candidates(v).name, ...
                {'global','current4','bands1_axial1_signed0','bands3_axial1_signed0'})
            cfg=r.cfg; cfg.do_reference_check=false;
            warm=helm_waveholtz_correction_rom(cfg,r); %#ok<NASGU>
            samples=zeros(5,numel(times));
            reduced_samples=samples; lift_samples=samples; completion_samples=samples;
            for rep=1:5
                repeated=helm_waveholtz_correction_rom(cfg,r);
                samples(rep,:)=repeated.test.online_total_time;
                reduced_samples(rep,:)=repeated.test.online_precorr_time;
                lift_samples(rep,:)=repeated.test.online_lift_time;
                completion_samples(rep,:)=repeated.test.online_fp_time;
            end
            times=mean(samples,1); maxsd=max(std(samples,0,1)); reps=5;
            save(fullfile(out,[candidates(v).name '_timings.mat']),'samples','reduced_samples','lift_samples','completion_samples','cfg');
        end
        s=rom_deployment_storage(r.correction_rom.layers); rho=zeros(numel(r.test.params),1);
        for k=1:numel(rho)
            A=r.problem.A_base+(1/r.test.params(k)^2-1)*r.problem.A_right;
            rho(k)=norm(r.rhs-A*r.test.online_final_u(:,k))/norm(r.rhs);
        end
        assert(all(rho<=1e-6),'PC true Helmholtz acceptance failed.');
        row=table(string(candidates(v).name),s.basis_MiB,s.payload_MiB, ...
            string(mat2str(s.ranks)),r.timing.offline,mean(times),max(times), ...
            maxsd,reps,mean(r.test.online_fp_iter),max(r.test.online_fp_iter), ...
            max(rho),max(r.test.online_precorr_relres_final),'VariableNames', ...
            {'candidate','basis_MiB','payload_MiB','ranks','offline_s', ...
            'online_mean_s','online_worst_s','max_query_timing_sd','repetitions', ...
            'completion_mean','completion_max','true_residual_max','precompletion_residual_max'});
        basebasis=row.basis_MiB; basepayload=row.payload_MiB;
        if ~isempty(summary), basebasis=summary.basis_MiB(1); basepayload=summary.payload_MiB(1); end
        row.basis_saving_percent=100*(1-row.basis_MiB/basebasis);
        row.payload_saving_percent=100*(1-row.payload_MiB/basepayload);
        summary=[summary;row]; %#ok<AGROW>
        writetable(summary,fullfile(out,['summary_' mode '.csv']));
    end
    disp(summary);
end

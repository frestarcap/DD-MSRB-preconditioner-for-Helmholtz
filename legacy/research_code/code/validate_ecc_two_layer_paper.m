function validate_ecc_two_layer_paper
% Check the publication cohort, serialized bases and repeated timing coverage.
    out=helm_output_path('outputs','ecc_two_layer_paper_20260911');
    s=load(fullfile(out,'paper_data.mat'),'global_results','dd_results');
    t=readtable(fullfile(out,'timings.csv'));
    storage=readtable(fullfile(out,'storage.csv'));
    assert(height(t)==99 && height(unique(t(:,{'method','query','repeat'})))==99);
    assert(all(t.final_update<1e-8) && all(t.true_residual<3e-10));
    assert(all(abs(t.total_seconds-t.rom_seconds-t.lift_seconds-t.completion_seconds)<1e-10));
    report=struct('passed',false,'timing_rows',height(t),'max_true_residual',max(t.true_residual), ...
        'max_prefix_error',max(t.prefix_error));
    for name={'global','dd'}
        r=s.([name{1} '_results']);
        assert(r.cfg.corr_num_layers==2 && r.correction_rom.num_layers==2);
        assert(isequal(r.cfg.corr_pod_singular_value_tols,[1e-3,1e-2]));
        m=load(fullfile(out,[name{1} '_lean_model.mat']),'model');model=m.model;
        w=whos('model');
        row=storage(strcmp(storage.method,name{1}),:);
        assert(abs(w.bytes/2^20-row.object_MiB)<1e-12);
        for j=1:2
            a=model.layers{j}.basis_space; b=r.correction_rom.layers{j}.basis_space;
            if strcmp(a.type,'dense'), assert(isequal(a.basis,b.basis));
            else
                assert(isequal(a.Un,b.Un) && isequal(a.Uf,b.Uf) && isequal(a.Up,b.Up));
                for field={'idx_near','idx_far','idx_pml'}
                    assert(isequal(a.partition.(field{1}),b.partition.(field{1})));
                end
            end
        end
        for k=1:numel(r.test.ratio)
            sys=r.test.systems{k}; u=r.test.online_final_u(:,k);
            rho=norm(sys.b-sys.A*u)/norm(sys.b);
            assert(rho<3e-10);
            h=r.test.online_precorr_relres_hist{k};assert(numel(h)==3);
            pre=norm(sys.b-sys.A*r.test.corr_u_layers{2}(:,k))/norm(sys.b);
            assert(abs(pre-h(end))<1e-12);
            q=t(strcmp(t.method,name{1}) & t.query==k,:);
            assert(height(q)==3 && numel(unique(q.completion_iters))==1);
            assert(q.completion_iters(1)==r.test.online_fp_iter(k));
            assert(abs(mean(q.total_seconds)-r.test.online_total_time(k))<1e-12);
        end
    end
    report.passed=true;
    f=fopen(fullfile(out,'validation.json'),'w');fprintf(f,'%s\n',jsonencode(report));fclose(f);
    fprintf('ECC_TWO_LAYER_VALIDATION_PASS\n');
end

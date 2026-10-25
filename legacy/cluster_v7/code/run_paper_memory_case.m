function run_paper_memory_case(problem,method,run_id)
% One fresh MATLAB process, one representative online solve (3D: warmup+solve).
    out=helm_output_path('outputs','paper_memory_benchmark');
    file=fullfile(out,[problem '_' method '.mat']);
    s=load(file,'cfg'); cfg=s.cfg; clear s;
    timer=tic;
    if strcmp(problem,'waveguide')
        result=run_jsc_3d_deployment(file,'lazy',0.317,1,false);
    else
        cfg.deployment_file=file;
        if strcmp(problem,'halfspace')
            cfg.test_params=0.862209674352;
            result=helm_waveholtz_correction_rom(cfg);
        else
            cfg.ratio_test=0.9072; cfg.ratio_test_auto_generated=false;
            result=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
        end
    end
    elapsed=toc(timer);
    s=load(file,'model'); model=s.model; clear s;
    info=whos('model'); model_bytes=info.bytes;
    save(fullfile(out,sprintf('%s_%s_run%d.mat',problem,method,run_id)), ...
        'result','elapsed','model_bytes');
    disp(result); fprintf('MODEL_BYTES %.0f\n',model_bytes);
end

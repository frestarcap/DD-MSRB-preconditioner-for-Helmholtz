function result = test_dd_refined_space_capture(outdir)
    if nargin<1
        outdir=fullfile(fileparts(fileparts(mfilename('fullpath'))),'outputs', ...
            'dd_highfreq_mechanism_20260913','refined_capture_test',char(datetime('now','Format','yyyyMMdd_HHmmssSSS')));
    end
    assert(~isfolder(outdir));mkdir(outdir);mkdir(fullfile(outdir,'vectors'));mkdir(fullfile(outdir,'reference'));
    training=fullfile(outdir,'training.mat');vectors=fullfile(outdir,'vectors');reference=fullfile(outdir,'reference');
    global_space=struct('type','dense','basis',[1 0;0 1;0 0;0 0]);
    blocks(1)=struct('basis',[1;0],'idx',[1;2]);blocks(2)=struct('basis',[1;0],'idx',[3;4]);
    dd_space=struct('type','blockn','blocks',blocks);
    models.global.layers={struct(),struct(),struct('space',global_space)};
    models.dd.layers={struct(),struct(),struct('space',dd_space)};
    resolved_problem_cfg=struct('omega',24);save(training,'models','resolved_problem_cfg');
    mu=.317;rho=1e-11;delta=[.1;0;0;0];save(fullfile(reference,'refined_reference.mat'),'mu','rho','delta');
    metadata=struct('source_training',training,'source_results','synthetic_original.mat', ...
        'target_met',true,'validated',true,'omega',24,'mu',mu);
    write_json(fullfile(reference,'summary.json'),metadata);write_json(fullfile(vectors,'complete.json'),metadata);
    for variants={'GG','DD','GGG','DDD'}
        label=variants{1};e0=[1;2;3;4];save(fullfile(vectors,sprintf('w24_mu0.317_%s_vectors.mat',label)),'label','mu','e0');
    end
    result=audit_dd_refined_space_capture(training,vectors,reference,fullfile(outdir,'capture.json'));
    assert(numel(result.targets)==4);
    assert(max([result.spaces.global.basis_gram_identity_deviation])==0);
    for k=1:4
        row=result.targets(k);
        assert(abs(row.global.captured_energy_fraction-5.21/30.21)<1e-14);
        assert(abs(row.dd.captured_energy_fraction-10.21/30.21)<1e-14);
        assert(row.global.orthogonal_energy_closure_error<1e-14);
        assert(row.dd.orthogonal_energy_closure_error<1e-14);
    end
    fprintf('DD_REFINED_CAPTURE_TEST_PASS\n');
end
function write_json(path,s)
    fid=fopen(path,'w');cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(s));
end

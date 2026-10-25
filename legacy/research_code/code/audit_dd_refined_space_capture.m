function report = audit_dd_refined_space_capture(training_file,vectors_dir,reference_dir,output_json)
%AUDIT_DD_REFINED_SPACE_CAPTURE Third-space Euclidean approximation of errors.
% Uses saved e0 + (refined reference - old reference), without A or solves.
% Independent best-space approximation is not the actual Galerkin correction.
    assert(~isfile(output_json),'Refusing to overwrite a space-capture audit.');
    reference_summary=jsondecode(fileread(fullfile(reference_dir,'summary.json')));
    completed=jsondecode(fileread(fullfile(vectors_dir,'complete.json')));
    assert(reference_summary.target_met && completed.validated);
    assert(strcmp(reference_summary.source_training,training_file));
    assert(strcmp(completed.source_training,training_file));
    assert(strcmp(reference_summary.source_results,completed.source_results), ...
        'The old references must originate in exactly the same saved result.');
    d=load(fullfile(reference_dir,'refined_reference.mat'),'delta','mu','rho');
    s=load(training_file,'models','resolved_problem_cfg');
    spaces=struct('global',s.models.global.layers{3}.space,'dd',s.models.dd.layers{3}.space);
    omega=s.resolved_problem_cfg.omega;clear s
    assert(omega==reference_summary.omega && abs(d.mu-reference_summary.mu)<1e-10);
    report=struct('definition_version','dd_refined_space_capture_v1', ...
        'training_file',training_file,'vectors_dir',vectors_dir,'reference_dir',reference_dir, ...
        'omega',omega,'mu',d.mu,'reference_true_residual',d.rho,'matlab_version',version, ...
        'error_definition','Saved old-reference e0 plus refined_reference.delta', ...
        'projection_definition','Best full-grid Euclidean orthogonal approximation in each fixed third-layer space; small Gram correction handles roundoff in basis orthogonality', ...
        'limitation','A stricter computed reference is not an error certificate; best Euclidean capture is not actual reduced Galerkin/FGMRES effectiveness');
    factors=struct();report.spaces=struct();
    for method={'global','dd'}
        m=method{1};[factors.(m),report.spaces.(m)]=factor_space(spaces.(m));
    end
    report.targets=struct([]);
    for variant={'GG','DD','GGG','DDD'}
        label=variant{1};file=fullfile(vectors_dir,sprintf('w%d_mu%.3f_%s_vectors.mat',omega,d.mu,label));
        old=load(file,'e0','mu','label');assert(strcmp(old.label,label) && abs(old.mu-d.mu)<1e-10);
        e=old.e0+d.delta;oldnorm=norm(old.e0);clear old
        en=norm(e);
        row=struct('variant',label,'source_vectors',file,'prefix_depth',numel(label), ...
            'old_reference_error_norm',oldnorm,'refined_error_norm',en, ...
            'reference_change_over_refined_error',norm(d.delta)/max(en,realmin));
        for method={'global','dd'}
            m=method{1};row.(m)=capture(spaces.(m),factors.(m),e);
        end
        if isempty(report.targets),report.targets=row;else,report.targets(end+1)=row;end
        fprintf('REFINED_CAPTURE omega=%g %s G=%.6f DD=%.6f\n', ...
            omega,label,row.global.captured_energy_fraction,row.dd.captured_energy_fraction);
    end
    report.validated=true;
    fid=fopen(output_json,'w');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end

function [f,meta]=factor_space(space)
    switch space.type
        case 'dense'
            bases={space.basis};
        case 'blockn'
            bases={space.blocks.basis};
        otherwise
            error('Unsupported saved space type: %s',space.type);
    end
    f=cell(size(bases));deviation=zeros(size(bases));rank=zeros(size(bases));
    for k=1:numel(bases)
        U=bases{k};rank(k)=size(U,2);
        if isempty(U),f{k}=[];continue;end
        H=U'*U;deviation(k)=norm(H-eye(size(H)),'fro');
        [R,flag]=chol((H+H')/2);
        assert(flag==0,'A saved basis has a rank-deficient Gram matrix.');
        f{k}=R;
    end
    meta=struct('space_type',space.type,'dimension',sum(rank),'block_ranks',rank, ...
        'basis_gram_identity_deviation',deviation);
end

function out=capture(space,f,e)
    total=norm(e)^2;captured=0;missed=0;
    if strcmp(space.type,'dense')
        [captured,missed]=one_block(space.basis,f{1},e);
    else
        for k=1:numel(space.blocks)
            [c,m]=one_block(space.blocks(k).basis,f{k},e(space.blocks(k).idx));
            captured=captured+c;missed=missed+m;
        end
    end
    closure=abs(captured+missed-total)/max(total,realmin);
    assert(closure<1e-9,'Orthogonal capture energy identity failed.');
    out=struct('captured_energy',captured,'unrepresented_energy',missed, ...
        'captured_energy_fraction',captured/max(total,realmin), ...
        'relative_best_approximation_error',sqrt(missed/max(total,realmin)), ...
        'orthogonal_energy_closure_error',closure);
end

function [captured,missed]=one_block(U,R,e)
    if isempty(U),captured=0;missed=norm(e)^2;return;end
    rhs=U'*e;coeff=R\(R'\rhs);projected=U*coeff;
    captured=norm(projected)^2;missed=norm(e-projected)^2;
end

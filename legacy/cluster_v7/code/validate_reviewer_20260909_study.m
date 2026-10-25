function validate_reviewer_20260909_study
%VALIDATE_REVIEWER_20260909_STUDY Independent algebra and source-support checks.
    out=helm_output_path('outputs','reviewer_20260909');
    for family={'frequency_1','two_parameter'}
        s=load(fullfile(out,family{1},'problem.mat'),'ops','b','c');
        mu=s.c.training(2,:); coeff=[1,mu(2)^(-2)-1];
        if numel(s.ops)==3, coeff(3)=mu(1)^(-2)-1; end
        A=s.ops{1};
        for i=2:numel(s.ops), A=A+coeff(i)*s.ops{i}; end
        for method={'global','dd'}
            d=load(fullfile(out,family{1},[method{1} '.mat']),'model','audit');
            Q=complex(zeros(numel(s.b),4)); Q(:,1)=s.b/norm(s.b);
            for j=1:3
                layer=d.model{j}; sp=layer.space; U=complex(zeros(sp.n,sp.k));
                for i=1:sp.k
                    unit=zeros(sp.k,1); unit(i)=1;
                    U(:,i)=rom_reduced_space_utils('lift',sp,unit);
                end
                assert(norm(U'*U-eye(sp.k),'fro')<1e-10);
                B=layer.operators{1};
                for i=2:numel(coeff), B=B+coeff(i)*layer.operators{i}; end
                assert(norm(B-U'*A*U,'fro')/max(norm(B,'fro'),eps)<1e-10);
                assert(norm(A*d.audit(j).E(:,2)-Q(:,j))<1e-9);
                dense_z=U*(B\(U'*Q(:,j)));
                z=rom_reduced_space_utils('lift',sp, ...
                    B\rom_reduced_space_utils('project',sp,Q(:,j)));
                assert(norm(z-dense_z)/max(norm(z),eps)<1e-10);
                [Q(:,j+1),~,~,stop]=rom_arnoldi_step(A*z,Q(:,1:j));
                if stop, break; end
            end
        end
    end
    blocks=struct('name',{'a','b'},'idx',{(1:2)',(3:4)'},'count',{2,2});
    part=struct('n',4,'blocks',blocks);
    cfg=struct('rank_global_tail_tol',1e-7,'rank_global_tail_strategy','marginal_storage');
    full=rom_reduced_space_utils('build_blockn_space',eye(4),part,cfg);
    assert(full.k==4,'Full local saturation must satisfy a positive budget.');
    cfg.rank_cap=1; rejected=false;
    try
        rom_reduced_space_utils('build_blockn_space',eye(4),part,cfg);
    catch e
        rejected=contains(e.message,'No feasible');
    end
    assert(rejected,'An impossible capped allocation must be rejected.');

    archive=helm_output_path('outputs','helm_trajectory_evidence', ...
        'review_repaired_20260905','ecc_residual_ablation_paper.mat');
    d=load(archive,'global_results'); r=d.global_results; clear d;
    systems=[r.train_reference.systems(:);r.test.systems(:)];
    rows=table();
    for k=1:numel(systems)
        sys=systems{k}; ratio=sys.ratio_ab;
        phys=false(r.problem.nx,r.problem.ny); phys(r.problem.I,r.problem.J)=true;
        forcing=sys.b; pml_norm=norm(forcing(~phys(:)));
        assert(pml_norm==0,'ECC boundary forcing must be zero in PML.');
        [ix,iy]=find(reshape(forcing~=0,size(phys)));
        xmin=min(r.problem.x_int(ix)); xmax=max(r.problem.x_int(ix));
        ymin=min(r.problem.y_int(iy)); ymax=max(r.problem.y_int(iy));
        rows=[rows;table(ratio,nnz(forcing),pml_norm,xmin,xmax,ymin,ymax, ...
            'VariableNames',{'ratio','forcing_nodes','PML_forcing_norm', ...
            'xmin','xmax','ymin','ymax'})]; %#ok<AGROW>
    end
    writetable(rows,fullfile(out,'ecc_forcing_support.csv'));
    fprintf('Reviewer algebra, saturation, and ECC forcing checks passed.\n');
end

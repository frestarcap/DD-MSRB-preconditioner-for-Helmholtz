function report = test_helm_3d_hpc()
%TEST_HELM_3D_HPC Compare optimized paths with the original coupled algebra.
    rng(82);
    p = helmholtz_3d_problem_interface(struct('Nphys', [9, 8, 7], ...
        'npml', [2, 2, 2], 'omega', 7, 'material_model', 'gaussian_lens'));
    raw = p.partition(.3);
    blocks = struct('name', {}, 'idx', {}, 'count', {});
    for j = 1:numel(raw.indices)
        blocks(j) = struct('name', sprintf('b%d',j), 'idx',raw.indices{j}, ...
            'count',numel(raw.indices{j}));
    end
    S = randn(p.N, 4) + 1i*randn(p.N, 4);
    space = rom_reduced_space_utils('build_blockn_space', S, ...
        struct('n',p.N,'blocks',blocks),struct('rank_rtol',1e-12));
    space.k=space.k-space.blocks(1).k;
    space.blocks(1).k=0;
    space.blocks(1).basis=complex(zeros(numel(space.blocks(1).idx),0));
    [R0,R1] = assemble_3d_rom_affine_sparse(p,space);
    T0 = rom_reduced_space_utils('assemble_reduced_matrix', ...
        @(v) p.K_pml*v+p.affine.mass_base*v,space);
    T1 = rom_reduced_space_utils('assemble_reduced_matrix', ...
        @(v) p.affine.mass_parameter_diagonal.*v,space);
    projection_error = max(norm(R0-T0,'fro')/norm(T0,'fro'), ...
        norm(R1-T1,'fro')/norm(T1,'fro'));
    assert(projection_error < 1e-12);
    [A,b] = p.assemble(.3);
    B = p.assemble_shifted(.3, .6);
    [M,~] = build_shifted_laplacian_multigrid_preconditioner(B,p.n, ...
        struct('mg_coarse_max_dofs',100));
    opts = struct('restart',20,'maxit',15,'tol',1e-9,'reorthogonalize',true);
    [x0,i0] = shifted_laplacian_gmres(@(v) A*v,M,b,[],opts);
    opts.orthogonalization='cgs2'; opts.store_preconditioned_basis=false;
    opts.fixed_linear_preconditioner=true;
    [x1,i1] = shifted_laplacian_gmres(@(v) A*v,M,b,[],opts);
    true_residual = norm(b-A*x1)/norm(b);
    difference = norm(x1-x0)/norm(x0);
    assert(i0.flag==0 && i1.flag==0 && true_residual<1.01e-9 && difference<1e-7);
    if exist('helm_gmres_cgs2_mex','file')==3
        for complex_basis=[false true]
            B=randn(50,9);
            if complex_basis, B=B+1i*randn(size(B)); end
            [B,~]=qr(B,0);
            for complex_rhs=[false true]
                v=randn(50,1);
                if complex_rhs, v=v+1i*randn(size(v)); end
                [wm,hm]=helm_gmres_cgs2_mex(B,v,6);
                h=B(:,1:6)'*v; wr=v-B(:,1:6)*h;
                h2=B(:,1:6)'*wr; wr=wr-B(:,1:6)*h2;
                assert(norm(wm-wr)/norm(v)<1e-12 && norm(hm-h-h2)/norm(v)<1e-12);
            end
        end
        opts.orthogonalization='cgs2_mex';
        [xm,im]=shifted_laplacian_gmres(@(v)A*v,M,b,[],opts);
        assert(im.flag==0 && norm(b-A*xm)/norm(b)<1.01e-9);
        assert(norm(xm-x1)/norm(x1)<1e-7);
        fprintf('CGS2_MEX_TEST_PASS difference=%.3e iterations=%d\n',norm(xm-x1)/norm(x1),im.iter_count);
    end
    report=struct('projection_error',projection_error,'solution_difference',difference, ...
        'true_residual',true_residual,'iterations',[i0.iter_count i1.iter_count]);
    disp(report); fprintf('HELM_3D_HPC_TEST_PASS\n');
end

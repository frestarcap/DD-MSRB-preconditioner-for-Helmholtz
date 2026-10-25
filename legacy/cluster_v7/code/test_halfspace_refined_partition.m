function test_halfspace_refined_partition()
%TEST_HALFSPACE_REFINED_PARTITION Coverage, geometry, and reduced algebra.
    cfg = struct('setup_only',true, 'Nphys_x',17, 'Nphys_y',13, ...
        'npml',4, 'omega',8, 'material_model','halfspace');
    r = helm_waveholtz_correction_rom(cfg);
    cfg = r.cfg;
    cfg.halfspace_dd_pml_mode = 'merged';
    part = build_halfspace_refined_partition(r.problem,cfg);
    [X,~] = ndgrid(r.problem.x_int,r.problem.y_int);
    physical = false(size(X));
    physical(r.problem.I,r.problem.J) = true;
    assert(isequal(part.blocks(1).idx,find(physical & X < cfg.interface_x)));
    assert(isequal(part.blocks(2).idx,find(physical & X >= cfg.interface_x)));
    assert(isequal(part.blocks(3).idx,find(~physical)));
    old_rng = rng;
    cleanup = onCleanup(@() rng(old_rng));
    rng(71);
    snapshots = randn(part.n,5) + 1i*randn(part.n,5);
    for physical_mode = {'slabs','source_rings'}
        for pml_mode = {'merged','signed','signed_depth', ...
                'material_signed_depth','material_symmetric_depth'}
            cfg.halfspace_dd_physical_mode = physical_mode{1};
            cfg.halfspace_dd_pml_mode = pml_mode{1};
            cfg.halfspace_dd_sections = 3;
            cfg.halfspace_dd_transverse_sections = 2;
            part = build_halfspace_refined_partition(r.problem,cfg);
            all_indices = vertcat(part.blocks.idx);
            assert(isequal(sort(all_indices),(1:part.n).'));
            for strategy = {'dof_weighted','marginal_storage'}
                trunc = struct('rank_global_tail_tol',0.2, ...
                    'rank_global_tail_strategy',strategy{1});
                space = rom_reduced_space_utils('build_block3_space',snapshots,part,trunc);
                B = zeros(space.n,space.k);
                for j = 1:space.k
                    e = zeros(space.k,1);
                    e(j) = 1;
                    B(:,j) = rom_reduced_space_utils('lift',space,e);
                end
                tail = norm(snapshots-B*(B'*snapshots),'fro')/norm(snapshots,'fro');
                assert(tail <= 0.2*(1+1e-12));
                A = r.problem.A_base;
                Ar = rom_reduced_space_utils('assemble_reduced_matrix',A,space);
                assert(norm(Ar-B'*A*B,'fro') <= 1e-11*norm(Ar,'fro'));
            end
        end
    end
    cfg.halfspace_dd_depth_edges = [0.7,0.3];
    rejected = false;
    try
        build_halfspace_refined_partition(r.problem,cfg);
    catch
        rejected = true;
    end
    assert(rejected);
    fprintf('Half-space refined partition tests passed.\n');
end

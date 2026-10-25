function test_block_reduced_assembly
% Compare explicit contraction with independent lifted and operator paths.
    old = rng; cleanup = onCleanup(@()rng(old)); %#ok<NASGU>
    rng(20260919); n = 43;
    A = sprandn(n,n,.2)+1i*sprandn(n,n,.2);
    indices = {[2:2:20].',[25:34].',[37:42].'};
    ranks = [3,0,2]; bases = cell(1,3);
    for j=1:3
        [bases{j},~] = qr(randn(numel(indices{j}),ranks(j))+1i*randn(numel(indices{j}),ranks(j)),0);
    end
    sp = struct('type','block3','n',n,'k',sum(ranks),'kn',ranks(1),'kf',ranks(2),'kp',ranks(3), ...
        'Un',bases{1},'Uf',bases{2},'Up',bases{3},'partition', ...
        struct('idx_near',indices{1},'idx_far',indices{2},'idx_pml',indices{3}));
    sn = struct('type','blockn','n',n,'k',sum(ranks));
    for j=1:3, sn.blocks(j)=struct('idx',indices{j},'basis',bases{j},'k',ranks(j)); end
    for item = {sp,sn}
        space=item{1}; B=zeros(n,space.k);
        for j=1:space.k
            e=zeros(space.k,1);e(j)=1;B(:,j)=rom_reduced_space_utils('lift',space,e);
        end
        ref=B'*A*B;
        for op={A,full(A),@(v)A*v}
            actual=rom_reduced_space_utils('assemble_reduced_matrix',op{1},space);
            assert(norm(actual-ref,'fro')<1e-12*max(1,norm(ref,'fro')));
        end
    end
    fprintf('BLOCK_REDUCED_ASSEMBLY_PASS\n');
end

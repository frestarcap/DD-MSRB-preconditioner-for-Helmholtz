function s=rom_deployment_storage(layers)
%ROM_DEPLOYMENT_STORAGE Numerical deployable payload, excluding MATLAB metadata.
% One shared index map is counted once. Training snapshots/dense diagnostic
% duplicates and full-order operators are not deployment basis storage.
    basis=0; operators=0; index=0; ranks=zeros(1,numel(layers));
    for j=1:numel(layers)
        l=layers{j}; sp=l.basis_space;
        ranks(j)=rom_reduced_space_utils('dimension',sp);
        if strcmp(sp.type,'dense')
            basis=basis+16*numel(sp.basis);
        elseif strcmp(sp.type,'blockn')
            basis=basis+16*sum(arrayfun(@(b)numel(b.basis),sp.blocks));
            index=max(index,8*sum([sp.blocks.count]));
        else
            basis=basis+16*(numel(sp.Un)+numel(sp.Uf)+numel(sp.Up));
            index=max(index,8*(size(sp.Un,1)+size(sp.Uf,1)+size(sp.Up,1)));
        end
        operators=operators+16*(numel(l.Ar_base)+numel(l.Ar_right));
    end
    s=struct('basis_MiB',basis/2^20,'operators_MiB',operators/2^20, ...
        'index_MiB',index/2^20,'payload_MiB',(basis+operators+index)/2^20,'ranks',ranks);
end

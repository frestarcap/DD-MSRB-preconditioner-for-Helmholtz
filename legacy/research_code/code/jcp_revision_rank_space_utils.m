function out = jcp_revision_rank_space_utils(action, varargin)
% Experiment-only rank controls. All nonexperimental operations use the
% active implementation. The shared ROM helper and its defaults are untouched.
    if ~strcmp(action,'build_block3_space') || numel(varargin)<3
        out=rom_reduced_space_utils(action,varargin{:}); return;
    end
    cfg=varargin{3};
    if ~isfield(cfg,'rank_global_tail_strategy') || ...
            ~ismember(cfg.rank_global_tail_strategy,{'uniform_relative','energy_greedy'})
        out=rom_reduced_space_utils(action,varargin{:}); return;
    end
    S=varargin{1}; partition=varargin{2}; eta=cfg.rank_global_tail_tol;
    assert(isfield(partition,'blocks') && ~isempty(partition.blocks));
    blocks=partition.blocks; nb=numel(blocks); U=cell(1,nb); sig=cell(1,nb);
    tails=cell(1,nb); energy=zeros(1,nb); ranks=zeros(1,nb);
    for b=1:nb
        [U{b},D,~]=svd(S(blocks(b).idx,:),'econ'); sig{b}=diag(D);
        energy(b)=sum(abs(sig{b}).^2);
        tails{b}=[flipud(cumsum(flipud(abs(sig{b}).^2)));0];
    end
    total=sum(energy); target=eta^2*total;
    switch cfg.rank_global_tail_strategy
        case 'uniform_relative'
            % Each block has relative tail <= eta. Summing squared tails
            % proves the same global relative Frobenius upper bound eta.
            for b=1:nb
                ranks(b)=find(tails{b}<=eta^2*energy(b)*(1+1e-12),1)-1;
            end
        case 'energy_greedy'
            % Largest next squared singular value: minimum total rank for
            % this separable aggregate error constraint, ignoring row cost.
            tail=total;
            while tail>target*(1+1e-12)
                score=-inf(1,nb);
                for b=1:nb
                    if ranks(b)<numel(sig{b})
                        score(b)=abs(sig{b}(ranks(b)+1))^2;
                    end
                end
                [best,b]=max(score); assert(isfinite(best));
                ranks(b)=ranks(b)+1;
                % Recompute remaining tail to avoid cancellation near 1e-5.
                tail=0;
                for q=1:nb, tail=tail+tails{q}(ranks(q)+1); end
            end
    end
    remaining=0;
    for b=1:nb, remaining=remaining+tails{b}(ranks(b)+1); end
    global_tail=sqrt(remaining/max(total,1e-30));
    assert(global_tail<=eta*(1+1e-10),'Global compression budget violated.');
    block_meta=struct(); names=cell(1,nb);
    for b=1:nb
        names{b}=char(blocks(b).name); local_tail=sqrt(tails{b}(ranks(b)+1)/max(energy(b),1e-30));
        meta=struct('nrow',numel(blocks(b).idx),'rank',ranks(b),'rank_numeric',ranks(b), ...
            'rank_cap',Inf,'rank_rtol',cfg.rank_rtol,'rank_tail_tol',[], ...
            'rank_rule',cfg.rank_global_tail_strategy,'singular_values',sig{b}, ...
            'rank_global_tail_tol',eta,'global_energy_weight',energy(b)/max(total,1e-30), ...
            'local_relative_tail',local_tail,'global_relative_tail_contribution', ...
            sqrt(energy(b)/max(total,1e-30))*local_tail,'global_relative_tail',global_tail, ...
            'generated_rank_tail_tol',local_tail);
        blocks(b).basis=U{b}(:,1:ranks(b)); blocks(b).k=ranks(b);
        blocks(b).meta=meta; blocks(b).label=names{b};
        block_meta.(matlab.lang.makeValidName(names{b}))=meta;
    end
    out=struct('type','blockn','n',size(S,1),'partition',partition, ...
        'blocks',blocks,'block_names',{names},'k',sum(ranks),'block_meta',block_meta);
end

function part = build_pc_refined_partition(prob,cfg)
%BUILD_PC_REFINED_PARTITION Geometry-only transverse bands and PML depth groups.
% Bands combine reflection-related rows. Axial cuts are optional; no test
% solution or test residual enters the partition.
    edges = option(cfg,'pc_dd_band_edges',[0.75,1.75,2.75]);
    sections = option(cfg,'pc_dd_axial_sections',1);
    depths = option(cfg,'pc_dd_depth_edges',0.5);
    signed = option(cfg,'pc_dd_signed_pml',false);
    validateattributes(sections,{'numeric'},{'scalar','integer','positive'});
    assert(all(diff(edges)>0) && all(edges>0) && all(edges<cfg.Ly/2));
    assert(all(diff(depths)>0) && all(depths>0) && all(depths<1));
    [X,Y] = ndgrid(prob.x_int,prob.y_int);
    [ii,jj] = ndgrid(1:prob.nx,1:prob.ny);
    physical = false(prob.nx,prob.ny); physical(prob.I,prob.J)=true;
    transverse = abs(Y-cfg.Ly/2);
    bands = [-Inf,edges,Inf]; axial = linspace(0,cfg.Lx,sections+1);
    blocks = struct('name',{},'idx',{},'count',{});
    for b=1:numel(bands)-1
        for a=1:sections
            mask=physical & transverse>bands(b) & transverse<=bands(b+1) & ...
                X>=axial(a) & X<=axial(a+1);
            if a>1, mask=mask & X>axial(a); end
            blocks=append(blocks,mask,sprintf('band%d_x%d',b,a));
        end
    end
    dx=max(max(prob.npml+1-ii,ii-(prob.nx-prob.npml)),0)/prob.npml;
    dy=max(max(prob.npml+1-jj,jj-(prob.ny-prob.npml)),0)/prob.npml;
    lr=dx>0 & dy==0; tb=~physical & ~lr;
    faces={lr,tb}; names={'lr','tb_corner'};
    ds=[0,depths,Inf]; depth=max(dx,dy);
    for f=1:2
        sides=1; if signed, sides=2; end
        for q=1:sides
            side=true(size(physical));
            if signed
                if f==1, side=X<cfg.Lx/2; else, side=Y<cfg.Ly/2; end
                if q==2, side=~side; end
            end
            for d=1:numel(ds)-1
                mask=faces{f} & side & depth>ds(d) & depth<=ds(d+1);
                blocks=append(blocks,mask,sprintf('%s_s%d_d%d',names{f},q,d));
            end
        end
    end
    ids=vertcat(blocks.idx);
    assert(numel(ids)==numel(X) && isequal(sort(ids),(1:numel(X)).'), ...
        'PC partition must cover each full-domain degree of freedom exactly once.');
    part=struct('nx',prob.nx,'ny',prob.ny,'n',numel(X), ...
        'mode','pc_refined','blocks',blocks);
end

function blocks=append(blocks,mask,name)
    idx=find(mask(:));
    if ~isempty(idx), blocks(end+1)=struct('name',name,'idx',idx,'count',numel(idx)); end
end

function value=option(cfg,name,fallback)
    value=fallback;
    if isfield(cfg,name), value=cfg.(name); end
end

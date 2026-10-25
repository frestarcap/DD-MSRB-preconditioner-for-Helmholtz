function plot_halfspace_dd_refinement(tag, selected)
%PLOT_HALFSPACE_DD_REFINEMENT Redraw stored payload and DD geometry diagnostics.
    if nargin < 1, tag = 'paper'; end
    if nargin < 2
        selected = {'global','material3-dof','slabs1-aligned-sym8-dof', ...
            'slabs2-aligned-sym8','slabs3-aligned-sym8'};
    end
    out = helm_output_path('outputs','halfspace_dd_refinement',char(tag));
    data = load(fullfile(out,'summary.mat'),'tbl');
    tbl = data.tbl;
    [present, idx] = ismember(string(selected), tbl.variant);
    assert(all(present), 'Some selected runs are missing.');
    tbl = tbl(idx,:);
    names = display_names(tbl.variant);
    fig = figure('Visible','off','Position',[80,80,1250,570],'Color','w');
    layout = tiledlayout(fig,1,2,'TileSpacing','compact');
    ax = nexttile(layout);
    h = barh(ax, [tbl.basis_bytes,tbl.operator_bytes,tbl.partition_bytes]/2^20, 'stacked');
    h(1).FaceColor = [0.05,0.45,0.25];
    h(2).FaceColor = [0.85,0.25,0.15];
    h(3).FaceColor = [0.5,0.5,0.5];
    set(ax,'YTick',1:height(tbl),'YTickLabel',names,'YDir','reverse', ...
        'FontSize',17,'TickLabelInterpreter','none');
    xlabel(ax,'ROM payload (MiB)');
    legend(ax,{'Basis','Reduced operators','Index map'},'Location','southoutside', ...
        'FontSize',15);
    grid(ax,'on');
    ax = nexttile(layout);
    plot(ax,tbl.mean_completion_iter,tbl.saved_basis_percent,'s', ...
        'Color',[0.05,0.45,0.25],'MarkerSize',10,'LineWidth',2);
    hold(ax,'on');
    plot(ax,tbl.mean_completion_iter,tbl.saved_rom_payload_percent,'o', ...
        'Color',[0.85,0.05,0.05],'MarkerSize',10,'LineWidth',2);
    set(ax,'FontSize',19);
    xlabel(ax,'Mean WaveHoltz completion iterations');
    ylabel(ax,'Storage saved vs global (%)');
    legend(ax,{'Basis','ROM payload'},'Location','southoutside','FontSize',16);
    grid(ax,'on'); box(ax,'on');
    xlim(ax,[min(tbl.mean_completion_iter)-1.5,max(tbl.mean_completion_iter)+1.5]);
    ylim(ax,[-2,max(tbl.saved_basis_percent)+4]);
    xlabel(ax,{'Mean WaveHoltz','completion iterations'});
    exportgraphics(fig,fullfile(out,'storage_tradeoff.png'),'Resolution',180);
    close(fig);
    geometry_names = selected(2:end);
    block_rows = cell(0,1);
    fig = figure('Visible','off','Position',[80,80,1250,660],'Color','w');
    layout = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    for k = 1:min(4,numel(geometry_names))
        saved = load(fullfile(out,[char(geometry_names{k}),'.mat']),'artifact');
        a = saved.artifact;
        for j = 1:numel(a.layers)
            d = a.layers{j}.diagnostics;
            for b = 1:numel(d.ranks)
                block_rows{end+1,1} = struct('variant',string(geometry_names{k}), ...
                    'layer',j,'block',d.labels(b),'dofs',d.dofs(b),'rank',d.ranks(b), ...
                    'basis_entries',d.dofs(b)*d.ranks(b), ...
                    'snapshot_energy_fraction',d.energy(b)/sum(d.energy), ...
                    'local_tail',d.local_tails(b),'global_tail',d.global_tail); %#ok<AGROW>
            end
        end
        p = a.layers{1}.basis_space.partition;
        labels = zeros(numel(a.x_int),numel(a.y_int));
        if isfield(p,'blocks')
            for b = 1:numel(p.blocks), labels(p.blocks(b).idx) = b; end
        else
            labels(p.idx_near) = 1;
            labels(p.idx_far) = 2;
            labels(p.idx_pml) = 3;
        end
        ax = nexttile(layout);
        imagesc(ax,a.x_int,a.y_int,labels.');
        set(ax,'YDir','normal','FontSize',17);
        count = max(labels(:));
        palette = hsv2rgb([mod((0:count-1)'*0.61803398875,1), ...
            0.48*ones(count,1),0.9*ones(count,1)]);
        colormap(ax,palette);
        hold(ax,'on');
        plot(ax,[0,a.cfg.Lx,a.cfg.Lx,0,0],[0,0,a.cfg.Ly,a.cfg.Ly,0], ...
            'k--','LineWidth',1.5);
        xline(ax,a.cfg.interface_x,'k-','LineWidth',1.2);
        plot(ax,a.x_int(a.physical_i(a.cfg.source_col_phys)), ...
            a.y_int(a.physical_j(a.cfg.source_row_phys)), ...
            'r*','MarkerSize',10,'LineWidth',2);
        axis(ax,'equal'); axis(ax,'tight');
        xlabel(ax,'x'); ylabel(ax,'y');
        title(ax,display_names(string(geometry_names{k})), ...
            'Interpreter','none','FontSize',17);
    end
    exportgraphics(fig,fullfile(out,'partitions.png'),'Resolution',180);
    close(fig);
    block_table = struct2table(vertcat(block_rows{:}));
    writetable(block_table,fullfile(out,'block_diagnostics.csv'));
end

function names = display_names(names)
    known = ["global","material3-dof","slabs1-aligned-sym8-dof", ...
        "slabs2-aligned-sym8","slabs3-aligned-sym8"];
    labels = ["Global","Original DD","Aligned DD (DOF)", ...
        "Aligned DD (4 strips)","Aligned DD (6 strips)"];
    [found,idx] = ismember(names,known);
    names(found) = labels(idx(found));
    names = replace(names,'-',' ');
end

function out_file = plot_helm_halfspace_dd_partition(output_dir,paper)
%PLOT_HELM_HALFSPACE_DD_PARTITION Draw the actual accepted ten-block masks.
    if nargin < 1 || isempty(output_dir)
        output_dir = helm_output_path('outputs','figures','helm_correction_rom');
    end
    if nargin < 2
        s = load(helm_output_path('outputs','halfspace_dd_paper','paper_data.mat'),'paper');
        paper = s.paper;
    end
    if ~exist(output_dir,'dir'), mkdir(output_dir); end
    part = paper.partition; a = paper.dd;
    assert(numel(part.blocks)==10 && strcmp(part.mode,'halfspace_refined'));
    pml_gray = (0.90:-0.10:0.20).';
    colors = [0.80 0.87 0.93;1 1 1;repmat(pml_gray,1,3)];
    fig = figure('Color','w','Position',[80 80 980 720],'Visible','off');
    ax = axes(fig);
    imagesc(ax,a.x_int,a.y_int,part.labels.'); axis(ax,'image');
    set(ax,'YDir','normal','FontName','Helvetica','FontSize',25,'LineWidth',1.6,'Box','on');
    colormap(ax,colors); clim(ax,[0.5 10.5]);
    xlabel(ax,'x'); ylabel(ax,'y'); title(ax,'Material- and depth-aligned DD');
    hold(ax,'on');
    rectangle(ax,'Position',[0 0 1 1],'LineWidth',1.6);
    plot(ax,[0.5 0.5],[0 1],'k--','LineWidth',1.4);
    for b = 1:10
        mask = part.labels==b;
        % Label the lower component when a block joins upper/lower strips.
        [ii,jj] = find(mask & ((1:part.ny) <= ceil(part.ny/2)));
        if isempty(ii), [ii,jj] = find(mask); end
        text(ax,mean(a.x_int(ii)),mean(a.y_int(jj)),num2str(b), ...
            'HorizontalAlignment','center','FontSize',24,'FontWeight','bold', ...
            'Color','k','BackgroundColor','w','Margin',1);
    end
    cb = colorbar(ax,'Ticks',1:10); cb.FontSize = 22; cb.Label.String = 'block index';
    out_file = fullfile(output_dir,'helm_halfspace_dd_partition.png');
    exportgraphics(fig,out_file,'Resolution',300); close(fig);
end

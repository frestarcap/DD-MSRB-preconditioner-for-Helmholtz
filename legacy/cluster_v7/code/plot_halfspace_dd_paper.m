function files = plot_halfspace_dd_paper(paper,c_target,target)
%PLOT_HALFSPACE_DD_PAPER Regenerate Experiment 1 figures from accepted MAT data.
    if nargin < 1 || isempty(paper)
        s = load(helm_output_path('outputs','halfspace_dd_paper','paper_data.mat'),'paper');
        paper = s.paper;
    end
    if nargin < 2 || isempty(c_target), c_target = 0.8622; end
    if nargin < 3 || isempty(target), target = 'all'; end
    target = validatestring(target, {'all','solution','residual'});
    out = helm_output_path('outputs','figures','helm_correction_rom');
    if ~exist(out,'dir'), mkdir(out); end
    g = paper.global; d = paper.dd; a = paper.aux;
    [~,k] = min(abs(g.test.params-c_target));
    c = g.test.params(k);
    tag = strrep(sprintf('c_right_%.4f',c),'.','_');
    red = [0.85 0.05 0.05]; green = [0.05 0.45 0.25];

    if ~strcmp(target,'solution')
        diagnostic_file=helm_output_path('outputs','halfspace_dd_paper','true_residual_representative.mat');
        if isfile(diagnostic_file) && abs(c-0.862209674352)<1e-10
            saved=load(diagnostic_file,'diagnostic');
            plot_halfspace_true_residual_figure(saved.diagnostic);
            files.residual=fullfile(out,['helm_trajectory_vs_nr_residual_decay_' tag '.png']);
        else
            fig = plot_rom_stage_residuals({g.test,d.test,a.nr},[k,k,k], ...
                {'Global TA-ROM','DD-TA-ROM','NR-ROM'},a.global_test,k, ...
                sprintf('Half-space, c_{right}=%.4f',c),'halfspace');
            files.residual = publish(fig,out,['helm_trajectory_vs_nr_residual_decay_' tag '.png'],true);
        end
    end
    if strcmp(target,'residual'), return; end

    fig = new_figure(980,720); ax = axes(fig);
    u = reshape(g.test.online_final_u(:,k),numel(g.x_int),numel(g.y_int));
    imagesc(ax,g.x_int,g.y_int,real(u.')); axis(ax,'image'); set(ax,'YDir','normal');
    colormap(ax,helm_solution_colormap(clim(ax))); cb = colorbar(ax); cb.FontSize = 23;
    style(ax,false); xlabel(ax,'x'); ylabel(ax,'y');
    title(ax,'Global TA-ROM: real(u)');
    files.solution = publish(fig,out,['helm_' tag '_real_joint_correction_full_field.png']);
    if strcmp(target,'solution'), return; end

    fig = new_figure(980,720); ax = axes(fig); hold(ax,'on');
    colors = lines(3);
    for j = 1:3
        sv = g.layers{j}.singular_values;
        semilogy(ax,1:numel(sv),max(sv,1e-16),'o-', ...
            'Color',colors(j,:),'LineWidth',3.2,'MarkerSize',9, ...
            'DisplayName',sprintf('layer %d',j));
    end
    style(ax); set(ax,'YScale','log'); xlabel(ax,'mode index'); ylabel(ax,'singular value');
    title(ax,'Training snapshot singular values'); legend(ax,'Location','northeast','FontSize',23);
    files.spectra = publish(fig,out,['helm_correction_rom_singular_values_' tag '.png']);

    files.partition = plot_helm_halfspace_dd_partition(out,paper);
    copyfile(files.partition,helm_output_path('LaTeX_DL_468198_240419','figures', ...
        'helm_correction_rom','helm_halfspace_dd_partition.png'));

    fig = new_figure(1340,580); tiles = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    ax = nexttile(tiles);
    bars = bar(ax,[paper.layer_table.global_entries,paper.layer_table.dd_entries]/1e3);
    bars(1).FaceColor = red; bars(2).FaceColor = green;
    style(ax); xlabel(ax,'layer'); ylabel(ax,'basis entries (10^3)');
    title(ax,'Layerwise basis storage');
    legend(ax,{'Global TA-ROM','DD-TA-ROM'},'Location','northwest','FontSize',20);
    ax = nexttile(tiles);
    physical = [sum(paper.layer_table.global_rank)*numel(g.physical_i)*numel(g.physical_j), ...
        sum(paper.block_table.dofs(1:2).*sum(paper.block_table{1:2,{'rank_1','rank_2','rank_3'}},2))];
    total = paper.main_rows.basis_bytes(:)'/16;
    bars = bar(ax,[physical;total-physical]'/1e3,'stacked');
    bars(1).FaceColor = [0.25 0.48 0.72]; bars(2).FaceColor = [0.75 0.75 0.75];
    style(ax); xticks(ax,1:2); xticklabels(ax,{'Global TA-ROM','DD-TA-ROM'});
    ylabel(ax,'basis entries (10^3)'); title(ax,'Physical / PML contributions');
    legend(ax,{'Physical','PML'},'Location','northeast','FontSize',20); ylim(ax,[0 1600]);
    files.storage = publish(fig,out,'helm_refined_dd_storage_paper.png');

    fig = new_figure(1340,580); tiles = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    ax = nexttile(tiles);
    values = reshape(paper.tolerance_rows.mean_iterations,2,[])';
    bars = bar(ax,values); bars(1).FaceColor = red; bars(2).FaceColor = green;
    style(ax); xticks(ax,1:3); xticklabels(ax,{'10^{-14}','10^{-13}','10^{-12}'});
    xlabel(ax,'WaveHoltz stopping tolerance'); ylabel(ax,'mean completion iterations');
    title(ax,'Completion updates'); ylim(ax,[0 27]);
    legend(ax,{'Global TA-ROM','DD-TA-ROM'},'Location','northeast','FontSize',20);
    ax = nexttile(tiles);
    values = reshape(paper.tolerance_rows.zero_update_count,2,[])';
    bars = bar(ax,values); bars(1).FaceColor = red; bars(2).FaceColor = green;
    style(ax); xticks(ax,1:3); xticklabels(ax,{'10^{-14}','10^{-13}','10^{-12}'});
    xlabel(ax,'WaveHoltz stopping tolerance'); ylabel(ax,'tests with zero WH updates');
    title(ax,'Zero-update queries'); ylim(ax,[0 25]);
    legend(ax,{'Global TA-ROM','DD-TA-ROM'},'Location','northeast','FontSize',20);
    files.tolerance = publish(fig,out,'helm_tol_tradeoff_paper.png');
end

function fig = new_figure(w,h)
    fig = figure('Color','w','Position',[80 80 w h],'Visible','off');
end

function style(ax,use_grid)
    if nargin < 2, use_grid = true; end
    set(ax,'FontName','Helvetica','FontSize',25,'LineWidth',1.6,'Box','on');
    ax.Title.FontSize = 27; ax.XLabel.FontSize = 28; ax.YLabel.FontSize = 28;
    if use_grid, grid(ax,'on'); ax.GridAlpha = 0.22; end
end

function curve(ax,y,line,color,label)
    semilogy(ax,0:numel(y)-1,max(y,1e-15),line,'Color',color, ...
        'LineWidth',3.2,'MarkerSize',9,'MarkerFaceColor','w','DisplayName',label);
end

function path = publish(fig,out,name,fixed_canvas)
    if nargin < 4, fixed_canvas = false; end
    path = fullfile(out,name);
    if fixed_canvas
        set(fig,'PaperPositionMode','auto');
        print(fig,path,'-dpng','-r300');
    else
        exportgraphics(fig,path,'Resolution',300);
    end
    copyfile(path,helm_output_path('LaTeX_DL_468198_240419','figures','helm_correction_rom',name));
    close(fig);
end

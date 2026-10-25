function files = plot_halfspace_dd_policy_residuals(tag, c_target)
%PLOT_HALFSPACE_DD_POLICY_RESIDUALS Redraw saved policy tests without solving.
    if nargin < 1, tag = 'policy_crosscheck_20260905'; end
    if nargin < 2, c_target = 0.8622; end
    assert(~isempty(regexp(tag, '^[A-Za-z0-9_-]+$', 'once')), 'Invalid tag.');
    source = helm_output_path('outputs','halfspace_dd_refinement',tag);
    out = helm_output_path('outputs','dd_rank_policy_comparison',tag,'figures');
    if ~exist(out,'dir'), mkdir(out); end
    names = {'global','slabs1-aligned-sym8-dof','slabs1-aligned-sym8'};
    labels = {'Global TA-ROM','DD: DOF-weighted','DD: marginal-storage greedy'};
    colors = [0.85 0.05 0.05; 0.05 0.45 0.25; 0.15 0.35 0.75];
    linespec = {'-o','-s','--d'};
    data = struct([]);
    for m = 1:3
        s = load(fullfile(source,[names{m} '.mat']),'artifact');
        t = s.artifact.test;
        data(m).params = t.params(:);
        data(m).completion = t.online_fp_relres_hist;
        data(m).iterations = t.online_fp_iter(:);
        data(m).tolerance = s.artifact.cfg.fp_tol;
        data(m).precorrection = cell(numel(t.params),1);
        for k = 1:numel(t.params)
            h = t.online_combined_relres_hist{k};
            n = t.online_combined_transition_idx(k);
            data(m).precorrection{k} = h(1:n);
            assert(numel(data(m).completion{k}) == data(m).iterations(k)+1);
            assert(abs(h(n)-t.online_precorr_relres_final(k)) < 1e-25);
        end
        assert(isequal(data(m).params,data(1).params), 'Test parameters differ.');
        clear s t;
    end
    [~, representative] = min(abs(data(1).params-c_target));
    [~, largest_gap] = max(data(3).iterations-data(2).iterations);
    files.representative = draw_case(representative,'representative');
    files.largest_gap = draw_case(largest_gap,'largest_iteration_gap');

    n = numel(data(1).params);
    stats = table(repmat(data(1).params,3,1),repelem(string(labels(:)),n), ...
        'VariableNames',{'c_right','method'});
    stats.helm_after_rom = zeros(3*n,1);
    stats.waveholtz_initial = zeros(3*n,1);
    stats.waveholtz_final = zeros(3*n,1);
    stats.completion_iterations = zeros(3*n,1);
    for m = 1:3
        rows = (m-1)*n+(1:n);
        stats.helm_after_rom(rows) = cellfun(@(h) h(end),data(m).precorrection);
        stats.waveholtz_initial(rows) = cellfun(@(h) h(1),data(m).completion);
        stats.waveholtz_final(rows) = cellfun(@(h) h(end),data(m).completion);
        stats.completion_iterations(rows) = data(m).iterations;
    end
    files.csv = fullfile(out,'halfspace_policy_residual_diagnostics.csv');
    writetable(stats,files.csv);
    save(fullfile(out,'halfspace_policy_residual_plot_data.mat'), ...
        'data','stats','representative','largest_gap','labels','colors','linespec');
    fprintf('Representative parameter: %.12f\n',data(1).params(representative));
    disp(stats(stats.c_right == data(1).params(representative),:));
    fprintf('Largest iteration gap parameter: %.12f\n',data(1).params(largest_gap));
    disp(stats(stats.c_right == data(1).params(largest_gap),:));
    fprintf('Greedy more/equal/fewer completion iterations: %d / %d / %d\n', ...
        nnz(data(3).iterations>data(2).iterations), ...
        nnz(data(3).iterations==data(2).iterations), ...
        nnz(data(3).iterations<data(2).iterations));
    disp(files);

    function path = draw_case(k, suffix)
        fig = figure('Visible','off','Color','w','Position',[60 60 1450 660]);
        tiles = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','loose');
        title(tiles,sprintf('Half-space, c_{right} = %.6f',data(1).params(k)), ...
            'FontName','Helvetica','FontSize',25,'FontWeight','normal');
        ax = nexttile(tiles); hold(ax,'on');
        for method = 1:3
            curve(ax,data(method).precorrection{k},method,labels{method});
        end
        style(ax); ylim(ax,[1e-16 2]); xticks(ax,0:3);
        yticks(ax,10.^(-16:4:0));
        xlabel(ax,'ROM-preconditioned FGMRES step');
        ylabel(ax,'Helmholtz relative residual');
        title(ax,'ROM precorrection');
        legend(ax,'Location','northeast','FontSize',18);
        ax = nexttile(tiles); hold(ax,'on');
        maxit = 0; initial = 0;
        for method = 1:3
            h = data(method).completion{k};
            count = data(method).iterations(k);
            curve(ax,h,method,sprintf('%s (%d it.)',labels{method},count));
            maxit = max(maxit,count); initial = max(initial,h(1));
        end
        yline(ax,data(1).tolerance,':','10^{-14}', ...
            'Color',[0.3 0.3 0.3],'FontSize',19,'LineWidth',1.5, ...
            'LabelHorizontalAlignment','left','HandleVisibility','off');
        style(ax); ylim(ax,[5e-15 10^(ceil(log10(initial))+1)]);
        xlim(ax,[0 maxit+1]);
        xlabel(ax,'WaveHoltz completion iteration');
        ylabel(ax,'WaveHoltz fixed-point relative residual');
        title(ax,'Completion from the ROM output');
        legend(ax,'Location','northeast','FontSize',17);
        path = fullfile(out,['halfspace_policy_residual_' suffix '.png']);
        exportgraphics(fig,path,'Resolution',180);
        savefig(fig,fullfile(out,['halfspace_policy_residual_' suffix '.fig']));
        close(fig);
    end

    function curve(ax,h,m,label)
        assert(all(isfinite(h)) && all(h>0), 'Invalid log-scale history.');
        semilogy(ax,0:numel(h)-1,h,linespec{m},'Color',colors(m,:), ...
            'LineWidth',2.7,'MarkerSize',7,'MarkerFaceColor','w', ...
            'DisplayName',label);
    end
end

function style(ax)
    set(ax,'FontName','Helvetica','FontSize',21,'LineWidth',1.3, ...
        'Box','on','YScale','log');
    grid(ax,'on'); ax.GridAlpha = 0.18;
    ax.Title.FontSize = 23; ax.Title.FontWeight = 'normal';
    ax.XLabel.FontSize = 22; ax.YLabel.FontSize = 22;
end

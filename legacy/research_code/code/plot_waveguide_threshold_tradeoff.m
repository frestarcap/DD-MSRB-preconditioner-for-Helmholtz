function comparison = plot_waveguide_threshold_tradeoff(out)
%PLOT_WAVEGUIDE_THRESHOLD_TRADEOFF Online speed versus persistent ROM storage.
    if nargin<1
        out=helm_output_path('outputs','waveguide_threshold_tradeoff_20260908','isolated_timing');
    end
    s=load(fullfile(out,'summary.mat'));
    g=sortrows(s.summary(s.summary.method=="global",:),'alpha');
    d=sortrows(s.summary(s.summary.method=="dd",:),'alpha');
    assert(isequal(g.alpha,d.alpha));
    point_ids="P"+string((1:height(g))');
    point_key=table(point_ids,g.eta1,g.eta2,'VariableNames',{'point_id','eta1','eta2'});
    writetable(point_key,fullfile(out,'threshold_point_key.csv'));
    saving=100*(1-d.payload_MiB./g.payload_MiB);
    relative_speed=g.online_mean_s./d.online_mean_s;
    comparison=table(g.alpha,g.eta1,g.eta2, ...
        g.payload_MiB,d.payload_MiB,saving, ...
        g.online_mean_s,d.online_mean_s,g.speedup,d.speedup,relative_speed, ...
        100*(1-d.online_mean_s./g.online_mean_s), ...
        'VariableNames',{'alpha','eta1','eta2','global_payload_MiB','dd_payload_MiB', ...
        'payload_saving_percent','global_online_s','dd_online_s', ...
        'global_speedup_vs_baseline','dd_speedup_vs_baseline', ...
        'dd_speedup_vs_global','dd_time_saving_vs_global_percent'});
    writetable(comparison,fullfile(out,'comparison.csv'));
    per_query=table();
    for k=1:height(g)
        for mu=unique(s.raw.parameter).'
            gt=s.raw(s.raw.alpha==g.alpha(k) & s.raw.method=="global" & s.raw.parameter==mu,:);
            dt=s.raw(s.raw.alpha==g.alpha(k) & s.raw.method=="dd" & s.raw.parameter==mu,:);
            bt=s.baseline(s.baseline.parameter==mu,:);
            row=table(g.alpha(k),mu,mean(bt.total_s),mean(gt.total_s),mean(dt.total_s), ...
                mean(bt.total_s)/mean(gt.total_s),mean(bt.total_s)/mean(dt.total_s), ...
                mean(gt.total_s)/mean(dt.total_s),mean(gt.completion_iterations), ...
                mean(dt.completion_iterations),max([gt.true_residual;dt.true_residual;bt.true_residual]), ...
                'VariableNames',{'alpha','parameter','baseline_s','global_s','dd_s', ...
                'global_speedup_vs_baseline','dd_speedup_vs_baseline','dd_speedup_vs_global', ...
                'global_completion_iterations','dd_completion_iterations','max_true_residual'});
            per_query=[per_query;row]; %#ok<AGROW>
        end
    end
    writetable(per_query,fullfile(out,'comparison_per_query.csv'));
    % Retain the same four displayed configurations; relabel original P2--P5 as P1--P4.
    show=point_ids~="P1";
    g=g(show,:);d=d(show,:);point_ids=point_ids(show);point_key=point_key(show,:);
    saving=saving(show);relative_speed=relative_speed(show);
    original_point_ids=point_ids;
    point_ids="P"+string((1:height(g))');
    point_key.point_id=point_ids;
    published_key=addvars(point_key,original_point_ids,'After','point_id');
    writetable(published_key,fullfile(out,'published_threshold_point_key.csv'));
    blue=[0.85,0.05,0.05]; orange=[0.05,0.45,0.25];
    fig=figure('Color','w','Position',[80,100,1320,650],'Visible','off');
    layout=tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    layout.OuterPosition=[0,0.18,1,0.82];
    ax=nexttile(layout); hold(ax,'on');
    plot(ax,g.payload_MiB,g.speedup,'-o','Color',blue,'LineWidth',2,'MarkerFaceColor','w','DisplayName','Global');
    plot(ax,d.payload_MiB,d.speedup,'-s','Color',orange,'LineWidth',2,'MarkerFaceColor','w','DisplayName','DD');
    yline(ax,1,'--','Color',[0.4,0.4,0.4],'DisplayName','MG-GMRES baseline');
    label_points(ax,g.payload_MiB,g.speedup,point_ids,blue,true);
    label_points(ax,d.payload_MiB,d.speedup,point_ids,orange,true);
    xlabel(ax,'Accounted ROM storage (MiB)'); ylabel(ax,'Online speedup over MG-GMRES');
    title(ax,'(a) Storage versus online acceleration'); legend(ax,'Location','best');
    ax.YLim=[0.9,max([g.speedup;d.speedup])*1.08];
    ax=nexttile(layout); hold(ax,'on');
    plot(ax,saving,relative_speed,'-s','Color',orange,'LineWidth',2,'MarkerFaceColor','w');
    yline(ax,1,'--','Equal online time','Color',[0.4,0.4,0.4],'LabelHorizontalAlignment','left');
    label_points(ax,saving,relative_speed,point_ids,orange,false);
    xlabel(ax,'DD ROM storage saving (%)'); ylabel(ax,'Global time / DD time');
    title(ax,'(b) DD compared with Global');
    ax.YLim=[min(relative_speed)-0.03,1.04];
    for ax=findall(fig,'Type','axes').'
        set(ax,'FontName','Helvetica','FontSize',17,'LineWidth',0.8,'Box','on');
        grid(ax,'on'); ax.GridAlpha=0.12;
        if strcmp(ax.XScale,'linear')
            ax.XLim=ax.XLim+[-1,1]*diff(ax.XLim)*0.06;
        end
    end
    add_threshold_key(fig,point_key);
    exportgraphics(fig,fullfile(out,'waveguide_threshold_tradeoff.png'),'Resolution',180);
    exportgraphics(fig,fullfile(out,'waveguide_threshold_tradeoff.pdf'),'ContentType','vector');
    for extension={'png','pdf'}
        name=['waveguide_threshold_tradeoff.' extension{1}];
        published=helm_output_path('LaTeX_DL_468198_240419','figures','helmholtz_3d',name);
        if isfile(published), copyfile(fullfile(out,name),published); end
    end
    close(fig);

    fig=figure('Color','w','Position',[80,100,1320,560],'Visible','off');
    layout=tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    layout.OuterPosition=[0,0.18,1,0.82];
    ax=nexttile(layout); hold(ax,'on');
    loglog(ax,g.alpha,g.precompletion_geomean,'-o','Color',blue,'LineWidth',1.6,'DisplayName','Global');
    loglog(ax,d.alpha,d.precompletion_geomean,'-s','Color',orange,'LineWidth',1.6,'DisplayName','DD');
    set(ax,'XScale','log','YScale','log');
    xlabel(ax,'Threshold setting'); ylabel(ax,'Post-ROM relative residual');
    title(ax,'Before completion (geometric mean)'); legend(ax,'Location','best');
    ax=nexttile(layout); hold(ax,'on');
    semilogx(ax,g.alpha,g.completion_mean,'-o','Color',blue,'LineWidth',1.6);
    semilogx(ax,d.alpha,d.completion_mean,'-s','Color',orange,'LineWidth',1.6);
    yline(ax,mean(s.baseline.completion_iterations),'--','Baseline');
    set(ax,'XScale','log');
    xlabel(ax,'Threshold setting'); ylabel(ax,'Mean completion iterations');
    title(ax,'Completion work to true residual 10^{-6}');
    for ax=findall(fig,'Type','axes').'
        set(ax,'FontName','Helvetica','FontSize',12,'Box','on', ...
            'XTick',g.alpha,'XTickLabel',point_ids,'XLim',[0.20,5]);
        grid(ax,'on'); ax.GridAlpha=0.12;
    end
    add_threshold_key(fig,point_key);
    exportgraphics(fig,fullfile(out,'waveguide_threshold_diagnostics.png'),'Resolution',180);
    exportgraphics(fig,fullfile(out,'waveguide_threshold_diagnostics.pdf'),'ContentType','vector');
    close(fig);
end

function label_points(ax,x,y,point_ids,color,prefer_left)
    used=false(size(point_ids));
    for k=1:numel(point_ids)
        if used(k), continue; end
        group=abs(x-x(k))<1e-9 & abs(y-y(k))<1e-9;
        used(group)=true;
        label=strjoin(point_ids(group),', ');
        span=max(max(x)-min(x),1);
        if prefer_left || x(k)>min(x)+0.95*span
            xpos=x(k)-0.02*span; alignment='right';
        else
            xpos=x(k)+0.02*span; alignment='left';
        end
        ypos=y(k)+0.045*max(max(y)-min(y),0.1);
        text(ax,xpos,ypos,label,'Color',color,'FontSize',13, ...
            'HorizontalAlignment',alignment,'VerticalAlignment','bottom', ...
            'Clipping','on','HandleVisibility','off');
    end
end

function add_threshold_key(fig,key)
    annotation(fig,'textbox',[0.05,0.105,0.9,0.052], ...
        'String','Threshold settings (\eta_1, \eta_2): P1 tighter \rightarrow P4 looser; P2 reference', ...
        'EdgeColor','none','HorizontalAlignment','center','FontSize',15,'Interpreter','tex');
    for k=1:height(key)
        annotation(fig,'textbox',[0.035+(k-1)*0.93/height(key),0.005,0.93/height(key),0.09], ...
            'String',sprintf('%s\n(%g, %g)',key.point_id(k),key.eta1(k),key.eta2(k)), ...
            'EdgeColor','none','HorizontalAlignment','center','FontSize',14,'Interpreter','none');
    end
end

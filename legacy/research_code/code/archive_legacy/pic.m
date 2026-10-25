% 数据（采样率 vs 误差）
sampling = [20, 10, 5];   % %
error = [5, 12, 40];        % %

% 画图
figure;
plot(sampling, error, '-o', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', 'r');

% 坐标轴设置
xlabel('Sampling Rate (%)', 'FontSize', 12);
ylabel('Error (%)', 'FontSize', 12);

% 让横轴从大到小（更符合"采样率降低"的直觉）
set(gca, 'XDir','reverse');

% 网格
grid on;

% 标注每个点（可选，但PPT很好用）
for i = 1:length(sampling)
    text(sampling(i), error(i), ...
        sprintf(' (%.0f%%, %.0f%%)', sampling(i), error(i)), ...
        'FontSize', 10);
end

% 标题（可选）
title('Trade-off: Sampling vs Error');
figure;

t = linspace(0, 2, 500);

initial_signal=squeeze(data_subset(1,1,:));
[imf,res]=vmd(initial_signal,'NumIMFs',8);

subplot(2, 5, 1);
plot(t, initial_signal');
set(gca, 'FontSize', 14);
% title('Original Signal','FontSize', 12);
title('原信号','FontSize', 14);

for i=2:9
    subplot(2, 5, i);
    plot(t, imf(:,(i-1)'));
    set(gca, 'FontSize', 14);
    title(['IMF ',num2str(i-1)],'FontSize', 14);
end

subplot(2, 5, 10);
plot(t, res');
set(gca, 'FontSize', 14);
% title('Residual','FontSize', 12);
title('残差','FontSize', 14);


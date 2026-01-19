clear,clc;
subj=1;
ses=1;
fs = 250;
corr_result = zeros(3,4);
ANOVA_result = zeros(3,4);
VMD_file_name = ['D:\Download\pyproject\testEEGNet\Data\19228725\VMD_files\subj_', num2str(subj),'_sess',num2str(ses),'_VMDdecomp.mat'];
fixed_bandwidth_file_name = ['D:\Download\pyproject\testEEGNet\Data\19228725\fixed_bandwidth\subj_', num2str(subj),'_sess',num2str(ses),'_decomp.mat'];
EMD_file_name = ['D:\Download\pyproject\testEEGNet\Data\19228725\EMD_files\subj_', num2str(subj),'_sess',num2str(ses),'_EMDdecomp.mat'];


for i=1:3
    if i==1
        load(VMD_file_name);
    elseif i==2
        load(fixed_bandwidth_file_name);
    elseif i==3
        load(EMD_file_name);
    end

    dsize = size(all_decomp_data);
    disp(dsize);
    
    IMFs=squeeze(all_decomp_data(1,1,:,:));

%     trasIMFS=abs(fft(IMFs));
%     trasIMFS=fft(IMFs);
%     
%     t = linspace(0, 2, 500);
%     
%     IMF_count=0;
%     trasIMFS_count =0;
%     f=(0:500-1) * fs / 500;
% 
%     for j=1:8
%         subplot(4, 2, j);
%         if mod(j,2)==0
%             trasIMFS_count = trasIMFS_count+1;
%             i_tras = fft(IMFs(:,trasIMFS_count),500);
%             plot(f, squeeze(trasIMFS(:,trasIMFS_count)));
%             title(['Spectrogram of IMF ',num2str(trasIMFS_count)],'FontSize', 12);
%         else
%             IMF_count=IMF_count+1;
%             plot(t, squeeze(IMFs(:,IMF_count)));
%             title(['IMF ',num2str(IMF_count)],'FontSize', 12);
%         end
%     end
        file_name = sprintf('sub-%03d_ses-%02d_task_motorimagery_eeg.mat', subj, ses);
        load(fullfile('D:\Download\pyproject\testEEGNet\Data\Shanghaidata',file_name));
        data_subset = data(:, :, 1:500);
        original_signal = squeeze(data_subset(1,1,:));

        for j=1:4
            corr_result(i,j)=corr(original_signal,IMFs(:,j));
%             ANOVA_result(i,j)=sum((original_signal-IMFs(:,j)).^2)/500;
            ANOVA_result(i,j)=var(IMFs(:,j));
        end
end
corr_result=round(corr_result, 4);
ANOVA_result=round(ANOVA_result, 4);
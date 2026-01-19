% FB means fixed bandwidth,固定频带
% 初步设定为4个频带，分别为：
% 8-12Hz，mu节律
% 12-18Hz
% 18-24Hz
% 24-30Hz，后三个为beta节律的分解，以提出更多的细节特征

clear,clc;
% 获取所有文件名
files = dir('D:\Download\pyproject\testEEGNet\Data\19228725\*.mat');
low_freq = [8,12,18,24];
high_freq = [12,18,24,30];
acc_result = zeros(25,5);
imf_num = 4;
for subj = 1:25
    for ses =1:5
        fixed_bandwidth_file_name = ['D:\Download\pyproject\testEEGNet\Data\19228725\fixed_bandwidth\subj_', num2str(subj),'_sess',num2str(ses),'_decomp.mat'];
        % 读取当前session的数据
        file_name = sprintf('sub-%03d_ses-%02d_task_motorimagery_eeg.mat', subj, ses);
        load(fullfile('D:\Download\pyproject\testEEGNet\Data\Shanghaidata',file_name));
        % 截取前500个点，即前5秒的数据
        data_subset = data(:, :, 1:500);
        % 获取数据的维度
        data_num_dimensions=size(data_subset);

        if ~exist(fixed_bandwidth_file_name,'file')
            % 对于所有数据，进行固定频带分解操作         

            all_decomp_data = zeros(data_num_dimensions(1),data_num_dimensions(2), data_num_dimensions(3), imf_num);
            result = cell(1, imf_num);
            for trial = 1:data_num_dimensions(1)
                trial_data = zeros(data_num_dimensions(2), data_num_dimensions(3), imf_num);
                for channel =1:data_num_dimensions(2)
                    for i =1:4
                        trial_data(channel,:,i)=bandpass_filter_within_range(data_subset(trial,channel,:),low_freq(i),high_freq(i),250);
                    end
                end
                all_decomp_data(trial,:,:,:)=trial_data;
                if mod(trial,10)==0
                    disp(['Subject ',num2str(subj),' Session ',num2str(ses),' trial',num2str(trial),'/',num2str(data_num_dimensions(1)),'Finished!']);
                end
            end
            save(fixed_bandwidth_file_name, 'all_decomp_data', 'labels');
        end
        load(fixed_bandwidth_file_name);

         % 存储到cell中
        for imf_ver = 1:imf_num
            result{imf_ver}=all_decomp_data(:,:,:,imf_ver);
        end

        % 计算csp协方差矩阵和特征矩阵
        % 最后得到的csp特征矩阵为一个1×8的cell,其中每个数据的shape为sample × 4
        csp_matrix_result = cell(1, imf_num);
        csp_feature_result = cell(1, imf_num);

        for i = 1:imf_num
            EEGSignal = struct('x',reshape(result{i},[data_num_dimensions(3),data_num_dimensions(2),data_num_dimensions(1)]),'y',labels);
            csp_matrix_result{i} = learnCSP(EEGSignal);
            csp_feature_result{i} = extractCSPFeatures(EEGSignal,csp_matrix_result{i},2);
        end

        % 获取imf_num个频带拼接起来的特征矩阵sample × imf_num*4
        combined_features_matrix = [];
        % 遍历 cell 数组的每个元素
        for i = 1:imf_num
            % 获取当前 cell 中的矩阵
            current_matrix = csp_feature_result{i};
            % 将当前矩阵与 combined_matrix 水平拼接
            if isempty(combined_features_matrix)
                combined_features_matrix = current_matrix;
            else
                combined_features_matrix = horzcat(combined_features_matrix, current_matrix);
            end
        end

        % 逐步回归挑选特征

        % [~,~,~,eval,~,~,~]=stepwisefit(combined_features_matrix, double(labels'),'display','off','inmodel',inmodel_para);
        [~,~,~,eval,~,~,~]=stepwisefit(combined_features_matrix, double(labels'),'display','off');
        final_index=find(eval);
        if isempty(final_index)
            final_index=1:(imf_num*4);
        end
        altered_combined_features_matrix = combined_features_matrix(:,final_index);
        csp_feature_result = altered_combined_features_matrix;
        X=csp_feature_result;
        Y=labels;
       
        % 创建 SVM 分类器
        SVMModel = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true);
        
        % 设置 5 折交叉验证
        CVSVMModel = crossval(SVMModel, 'KFold', 5);
        
        % 计算每折的预测标签
        predictedLabels = kfoldPredict(CVSVMModel);
        
        % 计算混淆矩阵
        confMat = confusionmat(Y, predictedLabels);
        
        % 提取TP, FP, FN, TN
        TP = confMat(2,2);
        FP = confMat(1,2);
        FN = confMat(2,1);
        TN = confMat(1,1);
        
        % 计算 Precision 和 Recall
        Precision = TP / (TP + FP);
        Recall = TP / (TP + FN);
        
        % 计算 F1 Score
        F1_score = 2 * (Precision * Recall) / (Precision + Recall);
    
        % 计算 Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN);


        acc_result(subj,ses) = accuracy;
        f1_result(subj,ses) = F1_score;

    end
end
% file_name = ['SH_fixed_bandwidth_withing_session_result.xlsx'];
% xlswrite(file_name,acc_result);
matrix = acc_result;

% 计算矩阵中所有值的平均值
overallMean = mean(matrix(:));

% 输出结果
disp(['The overall mean of all values in the matrix is: ', num2str(overallMean)]);

% 计算每行的最大值
rowMaxValues = max(matrix, [], 2); % 返回每行的最大值，得到 25x1 的向量

% 计算这些最大值的平均值
meanOfRowMaxValues = mean(rowMaxValues);

% 输出结果
disp(['The mean of the maximum values of each row is: ', num2str(meanOfRowMaxValues)]);

matrix = f1_result;

% 计算矩阵中所有值的平均值
overallMean = mean(matrix(:));

% 输出结果
disp(['The overall mean of all values in the matrix is: ', num2str(overallMean)]);

% 计算每行的最大值
rowMaxValues = max(matrix, [], 2); % 返回每行的最大值，得到 25x1 的向量

% 计算这些最大值的平均值
meanOfRowMaxValues = mean(rowMaxValues);

% 输出结果
disp(['The mean of the maximum values of each row is: ', num2str(meanOfRowMaxValues)]);
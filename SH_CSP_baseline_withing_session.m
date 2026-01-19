clear,clc;
% 获取所有文件名
files = dir('D:\Download\pyproject\testEEGNet\Data\19228725\*.mat');
acc_result = zeros(25,5);

for subj = 1:25
    for ses =1:5
        
        % 读取当前session的数据
        file_name = sprintf('sub-%03d_ses-%02d_task_motorimagery_eeg.mat', subj, ses);
        load(fullfile('D:\Download\pyproject\testEEGNet\Data\Shanghaidata',file_name));
        % 截取前500个点，即前5秒的数据
        data_subset = data(:, :, 1:500);
%         data_subset=data;
        % 获取数据的维度
        data_num_dimensions=size(data_subset);

        EEGSignal = struct('x',reshape(data_subset,[data_num_dimensions(3),data_num_dimensions(2),data_num_dimensions(1)]),'y',labels);
        csp_matrix_result = learnCSP(EEGSignal);
        csp_feature_result = extractCSPFeatures(EEGSignal,csp_matrix_result,2);

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
% file_name = ['SH_baseline_withing_session_result.xlsx'];
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
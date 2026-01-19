clear,clc
file_dir = 'D:\Download\pyproject\Creat_VMD_CSP\*.mat';
% 获取所有文件名
files = dir(file_dir);
acc_result = zeros(20,1);
f1_result = zeros(20,1);
for subj = 1:20
    load(fullfile(files(subj).folder,files(subj).name));
    data_num_dimensions = size(data);
    EEGSignal = struct('x',reshape(data,[data_num_dimensions(3),data_num_dimensions(2),data_num_dimensions(1)]),'y',labels);
    csp_matrix_result = learnCSP(EEGSignal);
    csp_feature_result = extractCSPFeatures(EEGSignal,csp_matrix_result,2);

%     % 创建5折交叉验证对象
%     cv = cvpartition(labels, 'KFold', 5);
%     
%     % 定义LDA分类器
%     ldaClassifier = fitcdiscr(csp_feature_result, labels);
%     
%     % 进行交叉验证
%     crossvalResult = crossval(ldaClassifier, 'cvpartition', cv);
%     
%     % 获取交叉验证的性能指标（替换为您的性能指标）
%     accuracy = 1 - kfoldLoss(crossvalResult, 'LossFun', 'ClassifError');
%     
%     disp(['baseline 5折交叉验证的准确率: ', num2str(accuracy)]);
% 
%     acc_result(subj,1) = accuracy;
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


    acc_result(subj,1) = accuracy;
    f1_result(subj,1) = F1_score;
end

% file_name = ['Physionet_CSP_baseline.xlsx'];
% xlswrite(file_name,acc_result);
matrix = acc_result;

% 计算矩阵中所有值的平均值
overallMean = mean(matrix(:));

% 输出结果
disp(['The overall mean of all values in the matrix is: ', num2str(overallMean)]);

matrix = f1_result;

% 计算矩阵中所有值的平均值
overallMean = mean(matrix(:));

% 输出结果
disp(['The overall mean of all values in the matrix is: ', num2str(overallMean)]);
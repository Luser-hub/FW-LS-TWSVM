clear,clc
file_dir = 'D:\Download\pyproject\Creat_VMD_CSP\*.mat';
% 获取所有文件名
files = dir(file_dir);
% acc_result = zeros(20,3);
acc_result = zeros(20,1);
f1_result = zeros(20,1);
% 设置VMD分解的imf个数
imf_num = 5;
for subj = 1:20
    load(fullfile(files(subj).folder,files(subj).name));
    VMD_file_name = ['D:\Download\pyproject\testEEGNet\Data\physionet\VMD_files\subj_', num2str(subj),'_imf',num2str(imf_num),'_VMDdecomp.mat'];
    % 获取数据的维度
    data_num_dimensions=size(data);
    if ~exist(VMD_file_name,'file')
        % 对于所有数据，进行VMD分解操作
        all_decomp_data = zeros(data_num_dimensions(1),data_num_dimensions(2), data_num_dimensions(3), imf_num);
        result = cell(1, imf_num);
        for trial = 1:data_num_dimensions(1)
            trial_data = zeros(data_num_dimensions(2), data_num_dimensions(3), imf_num);
            for channel =1:data_num_dimensions(2)
                [imf,res]=vmd(squeeze(data(trial,channel,:)),'NumIMFs',imf_num);
                trial_data(channel,:,:)=imf;
            end
            all_decomp_data(trial,:,:,:)=trial_data;
            if mod(trial,10)==0
                disp(['Subject ',num2str(subj),' trial',num2str(trial),'/',num2str(data_num_dimensions(1)),'Finished!']);
            end
        end
        save(VMD_file_name, 'all_decomp_data', 'labels');
    end
    load(VMD_file_name);

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
    
    % 获取8个频带拼接起来的特征矩阵sample × 32
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

%     [~,~,~,eval,~,~,~]=stepwisefit(combined_features_matrix, double(labels'),'display','off');
%     final_index=find(eval);
%     disp(final_index);
% 
%     
%     if isempty(final_index)
% %             [~,~,~,eval,~,~,~]=stepwisefit(combined_features_matrix, double(labels'),'display','off','inmodel',inmodel_para);
%         final_index=1:(imf_num*4);
%     end
    final_index=1:(imf_num*4);

    altered_combined_features_matrix = combined_features_matrix(:,final_index);
%     feature_result_name = ['D:\matlabLearning\feature\PY\PY_subj',num2str(subj),'feature.mat'];
%     % 定义划分比例
%     trainRatio = 0.8;
%     
%     % 创建交叉验证分割对象
%     cv = cvpartition(labels, 'Holdout', 1 - trainRatio);
%     
%     % 获取训练集和测试集的逻辑索引
%     trainIdx = training(cv);
%     testIdx = test(cv);
%     
%     % 划分数据和标签
%     trainData = altered_combined_features_matrix(trainIdx, :);
%     trainLabel = labels(trainIdx)+1;
%     
%     testData = altered_combined_features_matrix(testIdx, :);
%     testLabel = labels(testIdx)+1;
%     save(feature_result_name, 'trainData', 'trainLabel','testData','testLabel');
%     csp_feature_result= altered_combined_features_matrix;
%     
%     % 对更改后的数据进行五折交叉验证
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
%     altered_accuracy = 1 - kfoldLoss(crossvalResult, 'LossFun', 'ClassifError');
%     
%     disp(['VMD+逐步回归 5折交叉验证的准确率: ', num2str(altered_accuracy)]);
% 
%     acc_result(subj,1) = altered_accuracy;
% 
%     % 对未更改的数据进行五折交叉验证
%     cv = cvpartition(labels, 'KFold', 5);
% 
%     % 定义LDA分类器
%     ldaClassifier = fitcdiscr(combined_features_matrix, labels);
%     
%     % 进行交叉验证
%     crossvalResult = crossval(ldaClassifier, 'cvpartition', cv);
% 
%     % 获取交叉验证的性能指标（替换为您的性能指标）
%     accuracy = 1 - kfoldLoss(crossvalResult, 'LossFun', 'ClassifError');
%     
%     disp(['VMD 5折交叉验证的准确率: ', num2str(accuracy)]);
% 
%     acc_result(subj,2) = accuracy;
%     
% 
%         % 将数据分为训练集和测试集
%     cv = cvpartition(labels, 'Holdout', 0.3);
%     trainData = altered_combined_features_matrix(training(cv), :);
%     trainLabels = labels(1, training(cv));
%     testData = altered_combined_features_matrix(test(cv), :);
%     testLabels = labels(1, test(cv));
%     
%     % 创建SVM分类器
%     svmModel = fitcsvm(trainData, trainLabels, 'KernelFunction', 'linear', 'Standardize', true);
%     
%     % 预测测试集
%     predictions = predict(svmModel, testData);
%     
%     % 评估分类器性能
%     confMat = confusionmat(testLabels, predictions);
%     SVMaccuracy = sum(diag(confMat)) / sum(confMat(:));
%     
%     % 显示结果
% %     disp('混淆矩阵:');
% %     disp(confMat);
%     fprintf('SVM分类器准确率: %.2f%%\n', SVMaccuracy * 100);
%     acc_result(subj,3) = SVMaccuracy;
%     
% 
%     disp(['Subject ',num2str(subj),' Finished!']);
    X=altered_combined_features_matrix;
    Y=labels;
   
    % 创建 SVM 分类器
    SVMModel = fitcsvm(X, Y, 'KernelFunction', 'linear', 'Standardize', true);
%     ldaClassifier = fitcdiscr(combined_features_matrix, labels);
    
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
% file_name = ['physionet_result_imf',num2str(imf_num),'_withoutSW.xlsx'];
% xlswrite(file_name,acc_result);
% data = result{4};
% save('PYsubj1csp.mat','data','labels')
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
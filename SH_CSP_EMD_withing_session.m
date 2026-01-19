clear,clc;
% 获取所有文件名
files = dir('D:\Download\pyproject\testEEGNet\Data\19228725\*.mat');

acc_result = zeros(25,5);
f1_result = zeros(25,5);
for subj = 1:25
   for ses =1:5
        EMD_file_name = ['D:\Download\pyproject\testEEGNet\Data\19228725\EMD_files\subj_', num2str(subj),'_sess',num2str(ses),'_EMDdecomp.mat'];
        
        % 读取当前session的数据
        file_name = sprintf('sub-%03d_ses-%02d_task_motorimagery_eeg.mat', subj, ses);
        load(fullfile('D:\Download\pyproject\testEEGNet\Data\Shanghaidata',file_name));
        
        % 截取前500个点，即前5秒的数据
        data_subset = data(:, :, 1:500);
        % 获取数据的维度
        data_num_dimensions=size(data_subset);
        
        if ~exist(EMD_file_name,'file')
            % 对于所有数据，进行EMD分解操作

            %% 这一步是为了确定最小imf分解个数,保证数据大小的一致性
            min_imf = 100;
            for trial = 1:data_num_dimensions(1)
                trial_min_imf = 100;

                for channel =1:data_num_dimensions(2)
                    if mean(data_subset(trial,channel,:))<0.000001
                        epsilon = 1e-6;
                        data_subset(trial,channel,:) = data_subset(trial,channel,:) + epsilon * rand(size(data_subset(trial,channel,:)));
                    end

                    [imf,res]=emd(squeeze(data_subset(trial,channel,:)));
                    if isempty(imf)
                        data_subset(trial,channel,:) = zeros(1,1,500);
                        epsilon = 1e-6;
                        data_subset(trial,channel,:) = data_subset(trial,channel,:) + epsilon * rand(size(data_subset(trial,channel,:)));
                        [imf,res]=emd(squeeze(data_subset(trial,channel,:)));
                    end
                    
                    if size(imf,2)<trial_min_imf
                        trial_min_imf = size(imf,2);
                    end
                end

                if mod(trial,10)==0
                    disp(['Train : Subject ',num2str(subj),' Session ',num2str(ses),' trial',num2str(trial),'/',num2str(data_num_dimensions(1)),'Finished!']);
                end
                if trial_min_imf<min_imf
                    min_imf=trial_min_imf;
                end
            end
            % 最小imf必须为1
            if min_imf==0
                min_imf =1 ;
            end
            %%
            all_decomp_data = zeros(data_num_dimensions(1),data_num_dimensions(2), data_num_dimensions(3), min_imf);
            result = cell(1, min_imf);
            for trial = 1:data_num_dimensions(1)
                trial_data = zeros(data_num_dimensions(2), data_num_dimensions(3), min_imf);
                for channel =1:data_num_dimensions(2)
                    [imf,res]=emd(squeeze(data_subset(trial,channel,:)),'MaxNumIMF',min_imf);
                    trial_data(channel,:,:)=imf;
                end
                all_decomp_data(trial,:,:,:)=trial_data;
                if mod(trial,10)==0
                    disp(['Subject ',num2str(subj),' Session ',num2str(ses),' trial',num2str(trial),'/',num2str(data_num_dimensions(1)),'Finished!']);
                end
            end
            save(EMD_file_name, 'all_decomp_data', 'labels');
        end
        load(EMD_file_name);
        imf_num = size(all_decomp_data,4);
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
        csp_feature_result= altered_combined_features_matrix;
        
%         % 对未更改的数据进行五折交叉验证
%         cv = cvpartition(labels, 'KFold', 5);
%     
%         % 定义LDA分类器
%         ldaClassifier = fitcdiscr(combined_features_matrix, labels);
%         
%         % 进行交叉验证
%         crossvalResult = crossval(ldaClassifier, 'cvpartition', cv);
%     
%         % 获取交叉验证的性能指标（替换为您的性能指标）
%         acc = 1 - kfoldLoss(crossvalResult, 'LossFun', 'ClassifError');
%     
%         disp(['LDA分类器在测试集上的准确率: ', num2str(acc)]);
%         acc_result(subj,ses) = acc;
% 
%         disp(['Subject ',num2str(subj),' Session ',num2str(ses),' Finished!']);
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
% 
% file_name = ['SH_EMD_withing_session_result.xlsx'];
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
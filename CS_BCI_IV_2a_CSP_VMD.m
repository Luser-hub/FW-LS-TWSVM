clear,clc;
% 获取所有文件名
target_files = {'BCI-IV-2a_Subj1.mat', 'BCI-IV-2a_Subj2.mat', 'BCI-IV-2a_Subj3.mat', 'BCI-IV-2a_Subj4.mat',...
    'BCI-IV-2a_Subj5.mat', 'BCI-IV-2a_Subj6.mat', 'BCI-IV-2a_Subj7.mat', 'BCI-IV-2a_Subj8.mat','BCI-IV-2a_Subj9.mat'};
subj_list = { ...
    [16 15 10  9  8 14  4 11  7  3  2  5 19 17 13  1 18 20  6 12],...
[16 15 14 10  8  9  2  3  4 11  7 17 19 13  5 20],...
[ 8  9 10 16 14 15  4 11  2 19 17  3 13 18 20  5  7 12  1],...
[16 15 14 10  9  8  4  2  3  7 13 11 17 19  5  1 12  6 20 18 21],...
[ 8 16 15 14 10  9 11  4 17  2  3 19  7 13 20 18],...
[15  9 16 10  8 14  2  4  3 19 18 20 17  5 13  1  7 11 12],...
[10 16 15 14  8  9  3  4 17  2 19 11 20  7 13 18  5 12  1  6 21],...
[ 8 15 10  9 16 14  4  2  3 11  5 19  7  1 17 18],...
[10  9  8 15 16 14  4  2  3 19 20  5  1 18],...
};
% 每个被试
for subj = 1:9
    % 存储分解后的VMD文件
    VMD_file_name = ['E:\MIdata\BCI2a\VMD_files\subj_', num2str(subj),'_VMDdecomp.mat'];
    % 确定保留的特征个数
    numberOfFeatures = 16;
    if ~exist(VMD_file_name,'file')
        subj_data_array = [];
        subj_label_array = [];

        load(fullfile('D:\matlabLearning/',target_files{subj}));

        subj_data_array = data;
        subj_label_array = labels;
    
        % 获取数据和标签的维度
        data_num_dimensions = size(subj_data_array);
        label_num_dimensions = size(subj_label_array);
    
        % 对于所有数据，进行VMD分解操作
        all_decomp_data = zeros(data_num_dimensions(1),22, 875, 8);
        for iter = 1:data_num_dimensions(1)

            % 读取数据
            trial_data=squeeze(subj_data_array(iter,:,:));
            % 按照通道对数据进行VMD分解和CSP特征计算
            % VMD分解
            [channel,~]=size(trial_data);
            combined_data = zeros(22, 875, 8);
            for c =1:channel
                [imf,res]=vmd(trial_data(c,:),'NumIMFs',8);
                combined_data(c,:,:) = imf;
            end
            all_decomp_data(iter,:,:,:) = combined_data;
            disp(['subject ', num2str(subj), ' iter ', num2str(iter), ' / ', num2str(data_num_dimensions(1)), ' finished !']);   
        end
            save(VMD_file_name, 'all_decomp_data', 'subj_label_array');
    end
    load(VMD_file_name);
    
    % 选择当前 subj 对应的索引数组
    selected_channels = subj_list{subj};  % 取出当前遍历的索引数组
    
    % 选择 all_decomp_data 中相应的通道
    all_decomp_data = all_decomp_data(:, selected_channels, :, :); 

    all_data_shape = size(all_decomp_data);

    % 将8个频带分解的结果[sample × 32 × 500]放入到result中
    result = cell(1, 8);
    for i = 1:8
        result{i} = all_decomp_data(:,:,:,i);
    end
    
    % 进行多次采样，获取频带选取值
    % 存储结果
    spectral_result_index = zeros(8, 1);  
    result_index = zeros(32, 1);
    accs = 0;
%     for sample_loop = 1:50
%     if mod(sample_loop,10) ==0
%         disp(['----------- loop',num2str(sample_loop),'/50-----------'])
%     end
    % 采样一部分数据，获取采样的index
    % 生成随机排序的索引序列
%         sample_dim = all_data_shape(1);
%         random_indices = randperm(sample_dim);        


    % 计算csp协方差矩阵和特征矩阵
    % 最后得到的csp特征矩阵为一个1×8的cell,其中每个数据的shape为sample × 4
    csp_matrix_result = cell(4, 8);
    csp_feature_result = cell(4, 8);
    for i = 1:8
        for label_type = 1:4
            new_labels = subj_label_array; 
            new_labels(new_labels ~= label_type) = -1; 
            new_labels(new_labels == label_type) = 1; 
            EEGSignal = struct('x',reshape(result{i},[all_data_shape(3),all_data_shape(2),all_data_shape(1)]),'y',new_labels);
            csp_matrix_result{label_type,i} = learnCSP(EEGSignal);
            csp_feature_result{label_type,i} = extractCSPFeatures(EEGSignal,csp_matrix_result{i},2);
        end
    end


    % 预分配最终的 4×576×32 数组
    combined_features_matrix = zeros(4, 576, 32);
    
    % 遍历 4 个类别
    for row = 1:4
        % 将当前行 (8 个 cell) 合并为一个 576×32 矩阵
        merged_matrix = cell2mat(csp_feature_result(row, :));  % 结果 576×(4×8) = 576×32
        
        % 存入最终结果
        combined_features_matrix(row, :, :) = permute(merged_matrix, [3, 1, 2]);  % 变换维度为 4×576×32
    end
    
        % 使用逐步回归挑选特征
        % 获取所有系数值
%         swda = stepwisefit(combined_features_matrix, double(subj_label_array(random_indices(1:300))'),'PRemove',0.05);
%         % 对系数值取绝对值后进行排序,这样做是为了取对结果影响最大的列
%         [sorted_swda,index]=sort(abs(swda),'descend');
%         % 选取的特征数为numberOfFeatures
%         final_index = index(1:numberOfFeatures,:);


    N_features = 10;  % 你希望选取的特征个数
    altered_combined_features_matrix = cell(1,4);  % 存储 4 组特征
    final_indices_list = cell(1,4);  % 存储每个 label_type 的选中特征索引
    
    for label_type = 1:4
        % 取出数据并去掉第一维
        local_data = squeeze(combined_features_matrix(label_type, :, :));
    
        % 进行 stepwisefit 选择特征
        [B, SE, PVAL, eval, STATS, NEXTSTEP, HISTORY] = stepwisefit(local_data, double(subj_label_array'), ...
            'PRemove', 0.05, 'display', 'off');
    
        % 获取选中的特征索引
        feature_indices = find(eval);
    
        % 如果选择的特征多于 N_features，则取前 N_features 个
        if length(feature_indices) > N_features
            [~, sorted_idx] = sort(PVAL(feature_indices));  % 按 p 值排序
            feature_indices = feature_indices(sorted_idx(1:N_features));  % 仅保留前 N_features 个
        end
    
        % 存储该类别的特征索引
        final_indices_list{label_type} = feature_indices;
        
        % 存储该类别的特征数据
        altered_combined_features_matrix{label_type} = local_data(:, feature_indices);
    end
    
    %% **对齐所有特征维度**
    % 解决 `union()` 错误：手动展开 `cell array` 并合并
    common_indices = final_indices_list{1};  % 先取第一个类别的特征索引
    for i = 2:4
        common_indices = union(common_indices, final_indices_list{i}, 'sorted');  % 取并集
    end
    
    % 确保最终特征数不超过 N_features
    if length(common_indices) > N_features
        common_indices = common_indices(1:N_features);  % 仅保留前 N_features 个
    end
    
    % 重新筛选所有类别的数据，使得所有类别的最终特征数相同
    for label_type = 1:4
        local_data = squeeze(combined_features_matrix(label_type, :, :));
        altered_combined_features_matrix{label_type} = local_data(:, common_indices);
    end
    
    %% **合并所有处理后的数据**
    % 确定最终大小
    final_feature_dim = length(common_indices);
    altered_combined_features_matrix_final = zeros(4, 576, final_feature_dim);
    
    for label_type = 1:4
        altered_combined_features_matrix_final(label_type, :, :) = altered_combined_features_matrix{label_type};
    end
    
    %% **归一化**
    % 计算整个矩阵的 min 和 max 进行归一化
    minA = min(altered_combined_features_matrix_final, [], "all");
    maxA = max(altered_combined_features_matrix_final, [], "all");
    
    altered_combined_features_matrix_final = (altered_combined_features_matrix_final - minA) / (maxA - minA);
    
    % **最终输出**
    disp(size(altered_combined_features_matrix_final));  % 确保输出形状为 [4, 576, N_features]
    csp_feature_result = altered_combined_features_matrix_final;
    file_name = sprintf('BCI-IV-2a_subj%d.mat', subj);
    save(file_name, 'csp_feature_result', 'subj_label_array');
        % 划分训练测试集
%         random_indices = randperm(size(csp_feature_result,1));
%         ind = round(0.7 * size(random_indices,2)); %按比例分
%         randomized_data = csp_feature_result(random_indices,:);
%         randomized_label = EEGSignal.y(:,random_indices);
%         
%         trainData = randomized_data(1:ind,:); %训练集
%         testData = randomized_data(ind+1:end,:); %测试集
%         trainLabel = randomized_label(:,1:ind);
%         testLabel = randomized_label(:,ind+1:end);
%         
%         使用SVM
%         SVMModel = fitcsvm(randomized_data,randomized_label,'Standardize',true);
%         CVSVMModel = crossval(SVMModel);
%         classLoss = kfoldLoss(CVSVMModel);
%         acc = (1-classLoss)*100;
%         
%         ldaClassifier = fitcdiscr(trainData,trainLabel);
%         % 在测试集上进行预测
%         predictedLabels = predict(ldaClassifier, testData);
%         
%         % 计算准确率
%         acc = sum(predictedLabels == testLabel') / numel(testLabel);
%         disp(['SVM分类器在测试集上的准确率: ', num2str(acc)]);
%         if acc>accs
%             accs = acc;
%         end
%     end
%     disp(['SVM测试精度：',num2str(accs)]);



% %         创建result_index数组，初始化为零
% %         current_result_index = zeros(32, 1);   
% %         直接使用索引向量遍历
% %         for featureindex = final_index
% %             将对应位置的值加一
% %             current_result_index(featureindex) = current_result_index(featureindex) + 1;
% %         end
% %         
% %         统计频谱选择
% %         for spectral = 1:8
% %             currrent_choice = sum(current_result_index((spectral-1)*4+1:spectral*4));
% %             if currrent_choice>0
% %                 disp(strcat('选择了第',spectral,'个频谱。'));
% %                 disp(spectral);
% %             end
% %             spectral_result_index(spectral) = spectral_result_index(spectral)+currrent_choice;
% %         end
% %         
% %         result_index = result_index + current_result_index;
% % 
% % 
% %         投票结果统计
% %         disp(['final_index的大小为',size(final_index)]);
% %         disp(final_index);
% %     end
% %     bar(spectral_result_index);
% %     
% %     获取并去除最低的两个频带
% %     对向量进行排序
% %     [sorted_vector, sorted_indices] = sort(spectral_result_index);
% %     取最小的两个频带
% %     min_positions = sorted_indices(1:2);
% %     取最小的频带
% %     min_positions = sorted_indices(1);
% %     创造原始数据和调整后的容器
% %     all_data = zeros(size(current_train_data));
% %     altered_data = zeros(size(current_train_data));
% %     for spec = 1:8
% %         all_data = all_data+result{i};
% %         if ismember(spec, min_positions)
% %             continue;
% %         end
% %         altered_data=altered_data+result{i};
% %     end

    % 使用CSP+LDA进行对比试验
%     allEEGSignal = struct('x',reshape(all_data,[500,32,size(all_data,1)]),'y',subj_label_array(random_indices));
%     acc = calculate_CSP_LDA_ACC(allEEGSignal);
%     alteredEEGSignal = struct('x',reshape(altered_data,[500,32,size(altered_data,1)]),'y',subj_label_array(random_indices));
%     altered_acc = calculate_CSP_LDA_ACC(alteredEEGSignal);
% 
%     % 记录修改前后的精度提升
%     improvement = altered_acc-acc;

end

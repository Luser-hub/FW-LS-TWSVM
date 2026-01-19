% 获取所有文件名
target_files = {'file_a.mat', 'file_b.mat', 'file_f.mat', 'file_g.mat'};
% 每个被试
for subj = 1:4
    % 存储分解后的VMD文件
    VMD_file_name = ['E:\MIdata\BCI-IV-I\VMD_files\subj_', num2str(subj),'_VMDdecomp.mat'];
    % 确定保留的特征个数
    numberOfFeatures = 16;
    if ~exist(VMD_file_name,'file')
        subj_data_array = [];
        subj_label_array = [];
        file_name = sprintf('D:/Download/pyproject/testEEGNet/Data/BCI-IV-I/%s', target_files{subj});
        load(fullfile('D:\matlabLearning/',target_files{subj}));

        subj_data_array = data;
        subj_label_array = labels;
    
        % 获取数据和标签的维度
        data_num_dimensions = size(subj_data_array);
        label_num_dimensions = size(subj_label_array);
    
        % 对于所有数据，进行VMD分解操作
        all_decomp_data = zeros(data_num_dimensions(1),59, 400, 8);
        for iter = 1:data_num_dimensions(1)
            % 读取数据
            trial_data=squeeze(subj_data_array(iter,:,:));
            % 按照通道对数据进行VMD分解和CSP特征计算
            % VMD分解
            [channel,~]=size(trial_data);
            combined_data = zeros(59, 400, 8);
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
    for sample_loop = 1:50
        if mod(sample_loop,10) ==0
            disp(['----------- loop',num2str(sample_loop),'/50-----------'])
        end
        % 采样一部分数据，获取采样的index
        % 生成随机排序的索引序列
%         sample_dim = all_data_shape(1);
%         random_indices = randperm(sample_dim);        


        % 计算csp协方差矩阵和特征矩阵
        % 最后得到的csp特征矩阵为一个1×8的cell,其中每个数据的shape为sample × 4
        csp_matrix_result = cell(1, 8);
        csp_feature_result = cell(1, 8);
        for i = 1:8
            EEGSignal = struct('x',reshape(result{i},[400,59,all_data_shape(1)]),'y',subj_label_array);
%             current_train_data = result{i};
%             randomized_data = current_train_data(random_indices, :,: );
%             EEGSignal = struct('x',reshape(randomized_data,[500,32,300]),'y',subj_label_array(random_indices(1:300)));
            csp_matrix_result{i} = learnCSP(EEGSignal);
            csp_feature_result{i} = extractCSPFeatures(EEGSignal,csp_matrix_result{i},2);
        end
    
        % 获取8个频带拼接起来的特征矩阵sample × 32
        combined_features_matrix = [];
        % 遍历 cell 数组的每个元素
        for i = 1:8
            % 获取当前 cell 中的矩阵
            current_matrix = csp_feature_result{i};
            % 将当前矩阵与 combined_matrix 水平拼接
            if isempty(combined_features_matrix)
                combined_features_matrix = current_matrix;
            else
                combined_features_matrix = horzcat(combined_features_matrix, current_matrix);
            end
        end
    
        % 使用逐步回归挑选特征
        % 获取所有系数值
%         swda = stepwisefit(combined_features_matrix, double(subj_label_array(random_indices(1:300))'),'PRemove',0.05);
%         % 对系数值取绝对值后进行排序,这样做是为了取对结果影响最大的列
%         [sorted_swda,index]=sort(abs(swda),'descend');
%         % 选取的特征数为numberOfFeatures
%         final_index = index(1:numberOfFeatures,:);
        [~,~,~,eval,~,~,~]=stepwisefit(combined_features_matrix, double(subj_label_array'),'PRemove',0.05,'display','off');
        final_index=find(eval);
        altered_combined_features_matrix = combined_features_matrix(:,final_index);
        csp_feature_result= altered_combined_features_matrix;
        % 归一化
%         out = [];
%         n = size(csp_feature_result);
%         minA=min(csp_feature_result,[],"all");
%         maxA=max(csp_feature_result,[],"all");
%         csp_feature_result = (csp_feature_result-repmat(minA,n))./(maxA-minA);
        % 划分训练测试集
        random_indices = randperm(size(csp_feature_result,1));
        ind = round(0.7 * size(random_indices,2)); %按比例分
        randomized_data = csp_feature_result(random_indices,:);
        randomized_label = EEGSignal.y(:,random_indices);
        
        trainData = randomized_data(1:ind,:); %训练集
        testData = randomized_data(ind+1:end,:); %测试集
        trainLabel = randomized_label(:,1:ind);
        testLabel = randomized_label(:,ind+1:end);
        
        % 使用SVM
        SVMModel = fitcsvm(randomized_data,randomized_label,'Standardize',true);
        CVSVMModel = crossval(SVMModel);
        classLoss = kfoldLoss(CVSVMModel);
        acc = (1-classLoss)*100;
        
%         ldaClassifier = fitcdiscr(trainData,trainLabel);
%         % 在测试集上进行预测
%         predictedLabels = predict(ldaClassifier, testData);
%         
%         % 计算准确率
%         acc = sum(predictedLabels == testLabel') / numel(testLabel);
%         disp(['SVM分类器在测试集上的准确率: ', num2str(acc)]);
        if acc>accs
            accs = acc;
        end
    end
    disp(['SVM测试精度：',num2str(accs)]);



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

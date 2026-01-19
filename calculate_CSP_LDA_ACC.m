function acc=calculate_CSP_LDA_ACC(EEGSignal)
% 初始化
csp_matrix_result = learnCSP(EEGSignal);
csp_feature_result = extractCSPFeatures(EEGSignal,csp_matrix_result,2);
% 归一化
out = [];
n = size(csp_feature_result);
minA=min(csp_feature_result,[],"all");
maxA=max(csp_feature_result,[],"all");
csp_feature_result = (csp_feature_result-repmat(minA,n))./(maxA-minA);
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
% SVMModel = fitcsvm(randomized_data,randomized_label,'Standardize',true);
% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);
% acc = (1-classLoss)*100;

ldaClassifier = fitcdiscr(trainData,trainLabel);
% 在测试集上进行预测
predictedLabels = predict(ldaClassifier, testData);

% 计算准确率
acc = sum(predictedLabels == testLabel') / numel(testLabel);
disp(['LDA分类器在测试集上的准确率: ', num2str(acc)]);
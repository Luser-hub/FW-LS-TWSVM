function CSPMatrix = learnCSP(EEGSignals)

%输入
%EEGSignals: 需要训练的EEG信号，包括2类，其结构如下：
%    EEGSignals.x: EEG信号为[Ns*Nc*Nt]的三维矩阵形式，其中
%        Ns:每个trial的EEG样本数量
%        Nc:通道数量(EEG电极)
%        Nt: trials 的数量
%    EEGSignals.y: 一个[1*Nt]的向量包括每一类的类标签
%    EEGSignals.s: 采样率
%输出
%CSPMatrix:  一个学习到的CSP滤波器(一个[通道数量 * 通道数量] 的矩阵，用该矩阵的行作为滤波）

%初始化以及合法性检查
nbChannels = size(EEGSignals.x,2);        %通道大小
nbTrials = size(EEGSignals.x,3);          %trials大小
classLabels = unique(EEGSignals.y);       %获得类标签
nbClasses = length(classLabels);          %获取分类数
if nbClasses ~= 2                         %分类数大于2，报错！
    disp('ERROR! CSP can only be used for two classes');
    return;
end
covMatrices = cell(nbClasses,1);  %每一类的协方差矩阵

%计算每个trial标准化后的协方差矩阵
trialCov = zeros(nbChannels,nbChannels,nbTrials);
for t=1:nbTrials
    E = EEGSignals.x(:,:,t)';
    EE = E * E';
    trialCov(:,:,t) = EE ./ trace(EE);
end
clear E;
clear EE;
%%%%%%%%%%%%%%%%%%%%%%这里开始不同了%%%%%%%%%%%%%%%%%%%%%
%计算每一类的协方差矩阵
for c=1:nbClasses      
    covMatrices{c} = mean(trialCov(:,:,EEGSignals.y == classLabels(c)),3);  
end

%总的协方差矩阵
covTotal = covMatrices{1} + covMatrices{2};

%总协方差矩阵的白化转换
[Ut Dt] = eig(covTotal);     %注意：特征值初始是升序排列的

eigenvalues = diag(Dt);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
Ut = Ut(:,egIndex);
P = diag(sqrt(1./eigenvalues)) * Ut';

%通过P进行第一类协方差矩阵的转换
transformedCov1 =  P * covMatrices{1} * P';

%协方差矩阵的EVD
[U1 D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
U1 = U1(:, egIndex);
CSPMatrix = U1' * P;

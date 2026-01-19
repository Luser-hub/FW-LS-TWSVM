function features = extractCSPFeatures(EEGSignals, CSPMatrix, nbFilterPairs)
%Copyright (C) 2010  Fabien LOTTE
%输入
%EEGSignals: 需要训练的EEG信号，包括2类，其结构如下：
%    EEGSignals.x: EEG信号为[Ns*Nc*Nt]的三维矩阵形式，其中
%        Ns:每个trial的EEG样本数量
%        Nc:通道数量(EEG电极)
%        Nt: trials 的数量
%    EEGSignals.y: 一个[1*Nt]的向量包括每一类的类标签
%    EEGSignals.s: 采样率
%CSPMatrix: 学习到的CSP投影矩阵
%nbFilterPairs: CSP-m参数
%输出
%features: 从EEG数据集特征出来的特征, 是一个[Nt *(2*nbFilterPairs+1)]的矩阵，
%          最后一列为类标签

%初始化
nbTrials = size(EEGSignals.x,3);                                   %trial大小
features = zeros(nbTrials, 2*nbFilterPairs);                     %声明特征
Filter = CSPMatrix([1:nbFilterPairs (end-nbFilterPairs+1):end],:); %根据CSP-m参数确定滤波器大小

for t=1:nbTrials    
    %投影数据到CSP滤波器
    projectedTrial = Filter * EEGSignals.x(:,:,t)';    
    %投影信号的对数方差作为生成特征
    variances = var(projectedTrial,0,2);    
    for f=1:length(variances)
        features(t,f) = log(variances(f));
    end
%     features(t,end) = EEGSignals.y(t);                            %不要拼接标签    
end

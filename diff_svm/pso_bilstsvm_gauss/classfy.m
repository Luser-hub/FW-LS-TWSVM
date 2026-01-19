function [fenjie,probab]=classfy(C,ker,struct,test)
v1=struct.v1;
v2=struct.v2;
pp=size(v1,1);
data_predict1=myrbf_ker(test,C,ker);
 test_data1=[data_predict1,1];
 distance1=abs(test_data1*v1)/norm(v1(1:pp-1,:));           % 靠近正类点所在平面的预测结果
 distance2=abs(test_data1*v2)/norm(v2(1:pp-1,:));           % 靠近负类点所在平面的预测结果
 probab=[distance2/(distance1+distance2)  distance1/(distance1+distance2)];
 if distance1<distance2
       fenjie=1;                     % 将预测结果转化为标记为1与0的预测结果
    else
       fenjie=0;                    % 将预测结果转化为标记为1与0的预测结果
 end




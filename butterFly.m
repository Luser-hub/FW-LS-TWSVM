figure(2)
subplot(1,2,1); 
clear;
clc;
c = ['不感兴趣','觉得没必要','反正首页会刷到','其他'];
x = [1,2,3,4];
y = [21.43,30.95,57.14,7.14];
b=barh(x,y); %使用横向的坐标

set(gca,'YTickLabel',{'不感兴趣','觉得没必要','反正首页会刷到','其他'});
set(gca,'xticklabel',{'0%','10%','20%','30%','40%','50%','60%'});

%显示每个柱子上标注
xtips=b(1).XEndPoints;
ytips=b(1).YEndPoints;
label=string(b(1).YData)+'%';
text(ytips,xtips,label,'HorizontalAlignment','left')


subplot(1,2,2); 
clear;
clc;
yyaxis right;
c = ['不感兴趣','觉得没必要','反正首页会刷到','其他'];
x = [1,2,3,4];
y = [21.43,30.95,57.14,7.14];
b=barh(x,y); %使用横向的坐标

% set(gca,'YTickLabel',{'不感兴趣','觉得没必要','反正首页会刷到','其他'});
% set(gca,'xticklabel',{'0%','10%','20%','30%','40%','50%','60%'});
% 
% %显示每个柱子上标注
%  xtips=b(1).XEndPoints;
%  ytips=b(1).YEndPoints;
%  label=string(b(1).YData)+'%';
%  text(ytips,xtips,label,'HorizontalAlignment','left')


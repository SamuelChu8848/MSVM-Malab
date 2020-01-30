clear;clc;close all;
%% load Data;
data=xlsread('新提取表格.xls','1','A2:M6141');   % 共6140   （去除中间换泵、低效数据）低效数据定义为单耗高于103

% 提取某一时刻所有搭配
collocation=zeros(24);      %提取24个时刻内所有搭配
collocation_num=[];         %每个时刻泵组搭配的数量
for i=1:24
    k=1;
    for j=1:6140
        a=collocation(i,1:22);
        b=data(j,1);
        c=data(j,2);
        if(b==i&&~ismember(c,a))
            collocation(i,k)=data(j,2);
            k=k+1;
        end
    end
    collocation_num(end+1)=k-1;
end

%% 泵组搭配大循环
save myfile data collocation collocation_num;                   % 保存以上变量
for collocation_order=1:collocation_num                                        %【设置泵组循环】
% 提取给定时间、给定泵组搭配的数据
%上一个小时的5个数据作为输入 预测下一小时的6个数据
load myfile;                                                     %加载保存的数据
j=1;    %j为符合条件数据的条数。
timeset=xlsread('新预测操作.xls','工作台','C4:C4');                                   %【设置时刻：timeset】
matchset=collocation(timeset,collocation_order);                                                                       %【设置泵组搭配：matchset】
for i=1:6140
    time=data(i,1);
    match=data(i,2);
    if(time==timeset&&match==matchset)
        X(j,:)=data(i,3:7);
        Y(j,:)=data(i,8:end);
        j=j+1;
    end
end
%% 
[X_r,X_c]=size(X);          %如果某一时刻训练数据集数据不足10条，便提取所有时刻的特定搭配进行训练。
if (X_r<=10)
    j=1;
    for i=1:6140
        match=data(i,2);
        if(match==matchset)
            X(j,:)=data(i,3:7);
            Y(j,:)=data(i,8:end);
            j=j+1;
        end
    end
end

%% 归一化
[inputn,inputps]=mapminmax(X',0,1);
X=inputn';
[outputn,outputps]=mapminmax(Y',0,1);
Y=outputn';

%% 划分数据集
rand('state',0)
r=randperm(size(X,1));
divide = 0.8                                 %                                  【设置训练集与测试集的划分比例：divide】
ntrain =floor( size(X,1)*divide ) ;          % 划分训练集、测试集
Xtrain = X(r(1:ntrain),:);       % 训练集输入
Ytrain = Y(r(1:ntrain),:);       % 训练集输出
Xtest  = X(r(ntrain+1:end),:);   % 测试集输入
Ytest  = Y(r(ntrain+1:end),:);   % 测试集输出

[Ytest_r,Ytest_c]=size(Ytest)    % 测试集输出行数、列数（测试集的输出维度、测试条数）
[Ytrain_r,Ytrain_c]=size(Ytrain)    % 测试集输出行数、列数（测试集的输出维度、测试条数）

%% 没优化的msvm
% 随机产生惩罚参数与核参数
C    = 1000*rand;%惩罚参数
par  = 1000*rand;%核参数
ker  = 'rbf';
tol  = 1e-20;
epsi = 1;
% 训练
[Beta,NSV,Ktrain,i1] = msvr(Xtrain,Ytrain,ker,C,epsi,par,tol);
% 测试
Ktest = kernelmatrix(ker,Xtest',Xtrain',par);
Ypredtest = Ktest*Beta;

% 计算均方误差
mse_test=sum(sum((Ypredtest-Ytest).^2))/(size(Ytest,1)*size(Ytest,2))
 
% 反归一化
yuce=mapminmax('reverse',Ypredtest',outputps);yuce=yuce';
zhenshi=mapminmax('reverse',Ytest',outputps);zhenshi=zhenshi';

%% 粒子群优化多输出支持向量机
[y ,trace]=psoformsvm(Xtrain,Ytrain,Xtest,Ytest);
%% 利用得到最优惩罚参数与核参数重新训练一次支持向量机
C    = y(1);%惩罚参数
par  = y(2);%核参数
[Beta,NSV,Ktrain,i1] = msvr(Xtrain,Ytrain,ker,C,epsi,par,tol);
Ktest = kernelmatrix(ker,Xtest',Xtrain',par);
Ypredtest_pso = Ktest*Beta;
% 误差
pso_mse_test=sum(sum((Ypredtest_pso-Ytest).^2))/(size(Ytest,1)*size(Ytest,2))
% 反归一化 
yuce_pso=mapminmax('reverse',Ypredtest_pso',outputps);yuce_pso=yuce_pso';
subtraction=yuce_pso-zhenshi;

%% 部分变量字符化
match_char =num2str(matchset);      %将泵组搭配字符化
time_char  =num2str(timeset) ;      %将时刻字符化

sum_number=num2str(size(X,1));      %数据集总数字符化
divide_char=num2str(divide);        %将数据集划分比例字符化
Ytrain_r_char=num2str(Ytrain_r);    %训练集条数字符化
Ytest_r_char=num2str(Ytest_r);      %测试集条数字符化

%% 6个指标重合度计算                                                    【指标数量可修改】
repeat=zeros(1,Ytest_c);                                                        %【定义各指标重复度：repeat】
error =zeros(1,Ytest_c);
for i=1:Ytest_c
    for j=1:Ytest_r
        repeat(i)=repeat(i)+(zhenshi(j,i)-yuce_pso(j,i))^2;
        
    end
    repeat(i)=repeat(i)/Ytest_r;
    error(i) =max(abs(yuce_pso(:,i)-zhenshi(:,i)));         %误差绝对值的最大值
end

%% 预测结果数组
input_Xtest=xlsread('新预测操作.xls','工作台','D4:H4'); 
input_Xtest = [Xtest;input_Xtest];
Ktest = kernelmatrix(ker,input_Xtest',Xtrain',par);
Output_Ypredtest_pso = Ktest*Beta;
% 误差
Output_pso_mse_test=sum(sum((Ypredtest_pso-Ytest).^2))/(size(Ytest,1)*size(Ytest,2))
% 反归一化 
Output_yuce_pso=mapminmax('reverse',Output_Ypredtest_pso',outputps);Output_yuce_pso=Output_yuce_pso';
Output_yuce_pso = Output_yuce_pso (end,:);
%% 写入文件
cellnames_1=['C',num2str(collocation_order+7),':H',num2str(collocation_order+7)];
cellnames_2=['B',num2str(collocation_order+7),':B',num2str(collocation_order+7)];
cellnames_3=['I',num2str(collocation_order+7),':N',num2str(collocation_order+7)];
cellnames_4=['P',num2str(collocation_order+7),':U',num2str(collocation_order+7)];

xlswrite('新预测操作.xls',Output_yuce_pso,'工作台',cellnames_1)         %写入预测
xlswrite('新预测操作.xls',matchset,'工作台',cellnames_2)                %写入搭配
xlswrite('新预测操作.xls',error,'工作台',cellnames_3)                   %写入最大误差
xlswrite('新预测操作.xls',repeat,'工作台',cellnames_4)                  %写入重复度


%% 画图
% figure_name1=[time_char,'时刻,',match_char,'泵组搭配的','适应度寻优曲线'];
% figure('NumberTitle', 'off', 'Name',figure_name1);
% plot(trace)
% grid on;
% xlabel('迭代次数')
% ylabel('适应度值')
% title('psosvm适应度曲线（寻优曲线）')

%% 画图 将各个预测维度预测值与真实值的差做图像
figure_name2=[time_char,'时刻,',match_char,'泵组搭配的','测试集预测值与真实值各项指标的差值。其中数据集为',sum_number,'条,训练集为',Ytrain_r_char,'条，测试集为',Ytest_r_char,'条。'];
figure('NumberTitle', 'off', 'Name', figure_name2);
t=0:1:Ytest_r-1;axis([0 Ytest_r-1 -inf inf])
for i=1:Ytest_c
    subplot(2,3,i)
    hold on;grid on;
    plot(t,subtraction(:,i),'-r*')
    switch (i)
        case 1
            title('3#频率真实与预测');xlabel('X');ylabel('大小')
        case 2
            title('6#频率真实与预测');xlabel('X');ylabel('大小')
        case 3
            title('总浑水量真实与预测');xlabel('X');ylabel('大小')
        case 4
            title('浑水总压力真实与预测');xlabel('X');ylabel('大小')
        case 5
            title('一级电量真实与预测');xlabel('X');ylabel('大小')
        case 6
            title('一级单耗真实与预测');xlabel('X');ylabel('大小')
    end
end

%% 画图                                                               将各个预测维度与真实值对比，做出图像【2*3的图像】
figure_name3=[time_char,'时刻,',match_char,'泵组搭配的','测试集预测值与真实值对比。其中数据集为',sum_number,'条,训练集为',Ytrain_r_char,'条，测试集为',Ytest_r_char,'条。'];
figure('NumberTitle', 'off', 'Name', figure_name3);
t=0:1:Ytest_r-1;axis([0 Ytest_r-1 -inf inf])
for i=1:Ytest_c
    subplot(2,3,i)
    plot(t,zhenshi(:,i),'-bp')
    hold on;grid on;
    plot(t,yuce_pso(:,i),'-r*')
    switch (i)
        case 1
            legend('3#频率真实值','3#频率预测值','Location','SouthEast')
            title('3#频率真实与预测');xlabel('X');ylabel('大小')
        case 2
            legend('6#频率真实值','6#频率预测值','Location','SouthEast')
            title('6#频率真实与预测');xlabel('X');ylabel('大小')
        case 3
            legend('总浑水量真实值','总浑水量预测值','Location','SouthEast')
            title('总浑水量真实与预测');xlabel('X');ylabel('大小')
        case 4
            legend('浑水总压力真实值','浑水总压力预测值','Location','SouthEast')
            title('浑水总压力真实与预测');xlabel('X');ylabel('大小')
        case 5
            legend('一级电量真实值','一级电量预测值','Location','SouthEast')
            title('一级电量真实与预测');xlabel('X');ylabel('大小')
        case 6
            legend('一级单耗真实值','一级单耗预测值','Location','SouthEast')
            title('一级单耗真实与预测');xlabel('X');ylabel('大小')
    end
end     
clear;
end

save data_svm_psosvm 
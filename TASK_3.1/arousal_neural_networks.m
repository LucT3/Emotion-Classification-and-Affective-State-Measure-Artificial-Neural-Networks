%% TASK 3.1 (AROUSAL REGRESSION, MLP & RBF NETWORKS )


%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from data augmentation
data = load('data_augmentation_results\data_augmentation.mat');


%dataset for arousal
sample_arousal = data.sample_arousal;

%sorting
sample_arousal = sortrows(sample_arousal);


%plot AROUSAL classes distribution
t = sample_arousal(:,1);
edges = [0.75,1.25,2,2.5,3.5,4,4.75,5.25,6,6.5,7.5,8,8.75,9.25]
figure,histogram(t,'BinEdges',edges,'FaceColor','r')
title('AROUSAL values')
ylabel('values number')
xlabel('class')


%counter of datas
[numbs_arousal,val_arousal] = groupcounts(sample_arousal(:,1));


% data normalization
%sample_arousal = normalize(sample_arousal,'zscore'); %range and zscore tested

%dividing features and arousal values
X = sample_arousal(:,2:11);
t = sample_arousal(:,1);


% Data for regression
X_inv = X';
t_inv = t';


% dataset division, holdout partition
p=0.2;
%check if is correct
c = cvpartition(t,'Holdout',p);

idXTrain = training(c);
idXTest = test(c);
%features
X_train = X_inv(:,idXTrain);
X_test = X_inv(:,idXTest);

%aroual values
t_train = t_inv(:,idXTrain);
t_test = t_inv(:,idXTest);


%% MLP NETWORK FOR AROUSAL

net = fitnet(45);
% initial settings
net = init(net);
net.trainFcn = 'trainlm';
net.performFcn='mse';
net.trainParam.showCommandLine=1;
        
% Modify validation patience
net.trainParam.max_fail = 10;
        
% Dividing input data into training and validation
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;


%Train the Network
[net,tr] = train(net,X_train,t_train);

%Test the Network
y_test_arousal = net(X_test);
figure,plotregression(t_test, y_test_arousal,'Final REGRESSION reusults arousal ');

% View the Network
%view(net);

%plots
figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t_inv,y)


%% ---------------------------------------------- %%
%  ---------------------------------------------- %
%% AROUSAL RBF (Radial Basis Function Network)

%best result AROUSAL (spread= 1.83, MN =300, DF=50)
%Parameters for training
spread_arousal = 1.83;
goal_arousal = 0.0;
MN_arousal = 300;
DF_arousal = 50; 
rbf_net = newrb(X_train,t_train,goal_arousal, spread_arousal,MN_arousal, DF_arousal);

% Test RBF
output = rbf_net(X_test);
figure,plotregression(t_test, output, 'Final RBF reusults arousal');

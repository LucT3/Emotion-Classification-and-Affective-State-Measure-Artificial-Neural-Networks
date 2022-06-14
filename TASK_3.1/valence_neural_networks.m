%% TASK 3.1 (VALENCE REGRESSION, MLP & RBF NETWORKS )
 

%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from data augmentation
data =load('data_augmentation_results\data_augmentation_results.mat');


%dataset for valence
sample_valence = data.sample_valence;

%sorting
sample_valence = sortrows(sample_valence);


%plot AROUSAL classes distribution
t = sample_valence(:,1);
edges = [0.75,1.25,2,2.5,3.5,4,4.75,5.25,6,6.5,7.5,8,8.75,9.25]
figure,histogram(t,'BinEdges',edges)
title('VALENCE values')
ylabel('values number')
xlabel('class')

%counter of datas
[numbs_valence,val_valence] = groupcounts(sample_valence(:,1));


% data normalization
%sample_valence = normalize(sample_valence,'range'); %range and zscore tested

%dividing features and arousal values
X = sample_valence(:,2:11);
t = sample_valence(:,1);


% Data for regression
X_inv = X';
t_inv = t';


% dataset division, holdout partition
p=0.2;
c = cvpartition(t,'Holdout',p);

idXTrain = training(c);
idXTest = test(c);
%features
X_train = X_inv(:,idXTrain);
X_test = X_inv(:,idXTest);

%aroual values
t_train = t_inv(:,idXTrain);
t_test = t_inv(:,idXTest);


%% MLP NETWORK FOR VALENCE

%BEST RESULTS (neurons=20, maxfail=25, val= 0.2 train = 0.8)
net = fitnet(20);
% initial settings
net = init(net);
net.trainFcn = 'trainlm';
net.performFcn='mse';
net.trainParam.lr = 0.1; 
        
% Modify validation patience
net.trainParam.max_fail = 25; %tried with 5,10,15,20,25
net.trainParam.epochs = 300;
        
% Dividing input data into training and validation
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

net.trainParam.showCommandLine=1;


%Train the Network 
[net,tr] = train(net,X_train,t_train);

%Test the Network
y_test_valence = net(X_test);
figure,plotregression(t_test, y_test_valence,'Final REGRESSION reusults valence ');


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

%best result VALENCE: (spread= 1.05, MN =400, DF=50)
%Parameters for training
spread_valence = 1.05;
goal_valence = 0.0;
MN_valence = 400;
DF_valence = 50; 
rbf_net = newrb(X_train,t_train,goal_valence, spread_valence,MN_valence, DF_valence);

% Test RBF
output = rbf_net(X_test);
figure,plotregression(t_test, output, 'Final RBF reusults valence');

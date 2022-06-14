%% DATA PREPROCESSING (outliers, inf numb removals and feature selection)


%% Configuration
clear
close all
clc

%Config variables
times_sequentialfs = 5; %2,5 tested
outliers_removal_method = 'median';

%% Data Preprocessing 

% Removal of non numerical values
dataset = load('datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);
inf_val = isinf(dataset);
[rows_inf, col_inf] = find(inf_val == 1);
dataset(rows_inf,:) = [];

% Removal of outliers
dataset = dataset(:, 3:end);
dataset = rmoutliers(dataset, outliers_removal_method);


X = dataset(:,3:end);
t_arousal = dataset(:,1);
t_valence = dataset(:,2);


% Matrix to count how many times a specific feature has been selected by sequentialfs
features_arousal = [zeros(1,54); 1:54]';
features_valence = [zeros(1,54); 1:54]';



%% sequentialfs for AROUSAL

for j = 1:times_sequentialfs
    cv = cvpartition(t_arousal,'k',10);
    opt=statset('display','iter','UseParallel',true);
    [fs_arousal,history_arousal] =sequentialfs(@myfun,X,t_arousal,'cv',cv,'options',opt,'nfeatures',10);

    i = 1;    
    for val = fs_arousal
        if val == 1
            features_arousal(i, 1) = features_arousal(i, 1) + 1;
            fprintf("Added %d\n",i);
        end
        i = i+1;
    end

    %print values
    fprintf("-----------------------------------\n");
    fprintf("AROUSAL: "); 
    disp(features_arousal);
    fprintf("-----------------------------------\n");

    disp("Sorting...");
    features_arousal = sortrows(features_arousal, 1, 'descend');
    disp(features_arousal);

    disp(history_arousal);
end

%% RESULTS AROUSAL sequentialfs


%EXTRACTION METHOD 1 (USED METHOD)
%Select best 10 arousal features for task 3.1 (MLP)
features_arousal_best = features_arousal(1:10, 2);
best_arousal = [t_arousal X(:,features_arousal_best)];
%best 3 arousal features for task 3.3 (Fuzzy Inference System)
features_arousal_best3 = features_arousal(1:3, 2);
best3_arousal = [t_arousal X(:,features_arousal_best3)];


%EXTRACTION METHOD 2 (NOT USED RESULTS - current feature selection)
%best 10 features for task 3.1 (MLP)
arousal_res = X(:,fs_arousal);
arousal_res = [t_arousal arousal_res];
%best 3 features for task 3.3 (Fuzzy Inference System)
arousal_res_best3 = [t_arousal arousal_res(:,2:4)];



%% sequentialfs for VALENCE

for j = 1:times_sequentialfs
    cv = cvpartition(t_valence,'k',10);
    opt=statset('display','iter','UseParallel',true);
    [fs_valence,history_valence] =sequentialfs(@myfun,X,t_valence,'cv',cv,'options',opt,'nfeatures',10);

    i = 1;    
    for val = fs_valence
        if val == 1
            features_valence(i, 1) = features_valence(i, 1) + 1;
            fprintf("Added %d\n",i);
        end
        i = i+1;
    end

    %print values
    fprintf("-----------------------------------\n");
    fprintf("VALENCE: "); 
    disp(features_valence);
    fprintf("-----------------------------------\n");

    disp("Sorting...");
    features_valence = sortrows(features_valence, 1, 'descend');
    disp(features_valence);

    disp(history_valence);
end


%% RESULTS VALENCE sequentialfs


%EXTRACTION METHOD 1 (USED METHOD)
%best 10 valence features for task 3.1 (MLP)
features_valence_best = features_valence(1:10, 2);
best_valence = [t_valence X(:,features_valence_best)];
%best 3 valence features for task 3.3 (Fuzzy Inference System)
features_valence_best3 = features_valence(1:3, 2);
best3_valence = [t_valence X(:,features_valence_best3)];

 
%EXTRACTION METHOD 2 (NOT USED RESULTS - current feature selection) 
%best 10 features for task 3.1 (MLP)
valence_res = X(:,fs_valence);
valence_res = [t_valence valence_res];
%best 3 features for task 3.3 (Fuzzy Inference System)
valence_res_best3 = [t_valence valence_res(:,2:4)];



%% MYFUN for sequentialfs
%function for sequentialfs (creare script a parte)
function err = myfun(x_train, t_train, x_test, t_test)
%create a network
hiddenLayerSize=25;
net=fitnet(hiddenLayerSize);
net.trainParam.showWindow=0;

xx=x_train'; 
tt=t_train';

% train the network
[net,tr]=train(net,xx,tt);

% test the network
y=net( x_test');
err = perform(net, t_test',y);
end

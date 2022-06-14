%% TASK 3.3 (MAMDANI-TYPE FUZZY INFERENCE SYSTEM)


%% Preparation of data
clear
close all
clc


%Load results obtained from data augmentation
data = load('data_augmentation_results\data_augmentation_results.mat');

%dataset for arousal
sample_arousal = data.sample_arousal;

%counter of datas
[numbs_arousal,val_arousal] = groupcounts(sample_arousal(:,1));

%dividing AROUSAL features and values
X = sample_arousal(:,2:11);
t = sample_arousal(:,1);


%plot values for AROUSAL
edges = [0.75,1.25,2,2.5,3.5,4,4.75,5.25,6,6.5,7.5,8,8.75,9.25]
figure,histogram(t,'BinEdges',edges)
title('AROUSAL values')
ylabel('values number')
xlabel('class')


%get the best 3 AROUSAL features (selected in section 3.1)
best3_features = sample_arousal(:,2:4)';
best3_arousal_values = sample_arousal(:,1)';

p=0.3;
c = cvpartition(best3_arousal_values,'Holdout',p);
idxTrain = training(c);
X_train_arousal = best3_features(:,idxTrain);
t_train_arousal = best3_arousal_values(:,idxTrain);

idxTest = test(c);
X_test_arousal = best3_features(:,idxTest);
t_test_arousal = best3_arousal_values(:,idxTest);


%% DATA OBSERVATION (features distribution plot)

nbins = 15;
binWidth = 0.5;
binWidth2 = 0.005;
binWidth3 = 0.01;

x_values_1 = X_train_arousal(1, :);
x_values_2 = X_train_arousal(2, :);
x_values_3 = X_train_arousal(3, :);
y_values = t_train_arousal;

%ranges of values for best3 features
range_values_1 = [min(x_values_1) max(x_values_1)]
range_values_2 = [min(x_values_2) max(x_values_2)]
range_values_3 = [min(x_values_3) max(x_values_3)]


%Get indexes of samples for each specific output
y_indexes_one = find(y_values == val_arousal(1));
y_indexes_two = find(y_values == val_arousal(2));
y_indexes_three = find(y_values == val_arousal(3));
y_indexes_four = find(y_values == val_arousal(4));
y_indexes_five = find(y_values == val_arousal(5));
y_indexes_six = find(y_values == val_arousal(6));
y_indexes_seven = find(y_values == val_arousal(7));


%HISTOGRAMS TO DEFINE MEMBERSHIP FUNCTION
% Plot Histogram of features for a specific subset of outputs
figure
t = tiledlayout(3,3);
low_ind = [y_indexes_one y_indexes_two];
mid_ind = [y_indexes_three y_indexes_four y_indexes_five];
high_ind = [y_indexes_six y_indexes_seven];

nexttile
histogram(x_values_1(low_ind), 'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 1 when arousal: low [1-3)');
nexttile
histogram(x_values_2(low_ind),'BinWidth',binWidth2, 'BinLimits',[-0.05,0.1]);
title('Histogram of feature 2 when arousal: low [1-3)');
nexttile
histogram(x_values_3(low_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 3 when arousal: low [1-3)');

nexttile
histogram(x_values_1(mid_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 1 when arousal: mid [3-7]');
nexttile
histogram(x_values_2(mid_ind),'BinWidth',binWidth2, 'BinLimits',[-0.05,0.1]);
title('Histogram of feature 2 when arousal: mid [3-7]');
nexttile
histogram(x_values_3(mid_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 3 when arousal: mid [3-7]');

nexttile
histogram(x_values_1(high_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 1 when arousal: high (7-9]');
nexttile
histogram(x_values_2(high_ind), 'BinWidth',binWidth2, 'BinLimits',[-0.05,0.1]);
title('Histogram of feature 2 when arousal: high (7-9]');
nexttile
histogram(x_values_3(high_ind), 'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 3 when arousal: high (7-9]');


% HISTOGRAMS TO DEFINE RULES
% Plot Histogram of features for finding the empirical distribution
figure
t = tiledlayout(1,3);
nexttile
histogram(x_values_1,'BinWidth',binWidth);
title('Feature 1 distribution');
nexttile
histogram(x_values_2, 'BinWidth',binWidth3);
title('Feature 2 distribution');
nexttile
histogram(x_values_3, 'BinWidth',binWidth);
title('Feature 3 distribution');


%{
%% FIS TEST CREATION FROM COMMAND LINE 

%creation 
fis = mamfis("Name", "MamdaniFis");

%INPUT 1 
fis = addInput(fis,range_values_1,'Name',"x_values_1");
fis = addMF(fis,"x_values_1",'trapmf',[-8 -5 0.5 2.5],'Name',"Low");
fis = addMF(fis,"x_values_1",'trimf',[0.5 2.5 4.5],'Name',"Medium");
fis = addMF(fis,"x_values_1",'trapmf',[2.5 4.5 10 13],'Name',"High");

%INPUT 2
fis = addInput(fis,range_values_2,'Name',"feature_2");
fis = addMF(fis,"feature_2",'trapmf',[-0.02 -0.01 0.03 0.06],'Name',"Low");
fis = addMF(fis,"feature_2",'trimf',[0.03 0.06 0.09],'Name',"Medium");
fis = addMF(fis,"feature_2",'trapmf',[0.06 0.09 0.2 0.3],'Name',"High");

%INPUT 3
fis = addInput(fis,range_values_3,'Name', "feature_3");
fis = addMF(fis,"feature_3",'trapmf',[-8 -5 2.5 5],'Name',"Low");
fis = addMF(fis,"feature_3",'trimf',[2.5 5 7],'Name',"Medium");
fis = addMF(fis,"feature_3",'trapmf',[5 7 10 12],'Name',"High");

%Output configuration
fis = addOutput(fis,[1 9],'Name',"Arousal");
fis = addMF(fis,"Arousal","trimf",[-2.33 1 3.5],'Name',"Low");
fis = addMF(fis,"Arousal","trimf",[2.5 5 7.5],'Name',"Medium");
fis = addMF(fis,"Arousal","trimf",[6.5 9 12.3],'Name',"High");

%Rules configuration
ruleList = [
            ];

fis = addRule(fis,ruleList);
%}
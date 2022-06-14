%% DATA AUGMENTATION 

%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from sequential feature selection
data = load('data_preprocessing_results\data_preprocessing_results.mat');

%dataset for arousal and valence
sample_arousal = data.best_arousal;
sample_valence = data.best_valence;

%sorting
sample_arousal = sortrows(sample_arousal);
sample_valence = sortrows(sample_valence);

%original sets (for comparation)
sample_arousal_original = sample_arousal;
sample_valence_original = sample_valence;


%% DATA AUGMENTATION PARAMETERS

%counter of datas
[numbs_arousal,val_arousal] = groupcounts(sample_arousal(:,1));
[numbs_valence,val_valence] = groupcounts(sample_valence(:,1));

%number of desired values for each class
limit_value = 300;

%network parameters
hiddenSize = 45;
epochs= 1000;
i =1;

fprintf("SUMMARY:\n maximum class values: %d\n neurons number: %d\n epochs numbers: %d\n",limit_value,hiddenSize,epochs);



%% AROUSAL DATA AUGMENTATION 

%calculating augmentation iteration times for each class
for value =  1:length(numbs_arousal)
    limit_value_class(value) = round(limit_value/numbs_arousal(value));
end

limit_value_class
numbs_arousal

%dataset indexes
prev = -1;
next = -1;
flag = 1;
fprintf(' %d %d \n',prev,next);
for cursor = 1:(length(numbs_arousal))
    if (flag==1)
        prev =1;
        next = numbs_arousal(1);
        fprintf(' %d %d \n',prev,next);
        flag = 0;
        prova = sample_arousal(prev:next,1:end);
    else
        prev = next+1;
        next = (prev-1) + numbs_arousal(cursor);
        fprintf(' %d %d \n',prev,next);
    end
    
    %perform data augmentation
    disp('processing data augmentation...');
    disp(limit_value_class(cursor));
    while i<limit_value_class(cursor)
        autoenc = trainAutoencoder(sample_arousal(prev:next,2:end)',hiddenSize,...
            'EncoderTransferFunction','satlin',...
            'DecoderTransferFunction','purelin',...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',0,... 
            'ShowProgressWindow',false,...
            'SparsityProportion',1);
        reconstructed_datasample = predict(autoenc,sample_arousal(prev:next,2:end)');
        arousal_values = sample_arousal(prev:next,1);
        reconstructed_datasample = cat(1,arousal_values',reconstructed_datasample);
        sample_arousal = cat(1,sample_arousal,reconstructed_datasample');
        i=i+1; 
        
            
    end
    i=1;
end
%}



%% VALENCE DATA AUGMENTATION

%calculating augmentation iteration times for each class 
limit_value_class = [];
for value =  1:length(numbs_valence)
    limit_value_class(value) = round(limit_value/numbs_valence(value));
end

limit_value_class
numbs_valence

%dataset indexes 
prev = -1;
next = -1;
flag = 1;
fprintf(' %d %d \n',prev,next);

for cursor = 1:(length(numbs_valence))
    if (flag==1)
        prev =1;
        next = numbs_valence(1);
        fprintf(' %d %d \n',prev,next);
        flag = 0;
        prova = sample_valence(prev:next,1:end);
    else
        prev = next+1;
        next = (prev-1) + numbs_valence(cursor);
        fprintf(' %d %d \n',prev,next);
    end
    
    %perform data augmentation
    disp('processing data augmentation...');
    disp(limit_value_class(cursor));
    while i<limit_value_class(cursor)
        autoenc = trainAutoencoder(sample_valence(prev:next,2:end)',hiddenSize,...
            'EncoderTransferFunction','satlin',...
            'DecoderTransferFunction','purelin',...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',0,... 
            'ShowProgressWindow',false,...
            'SparsityProportion',1);
        
        reconstructed_datasample = predict(autoenc,sample_valence(prev:next,2:end)');
        valence_values = sample_valence(prev:next,1);
        reconstructed_datasample = cat(1,valence_values',reconstructed_datasample);
        sample_valence = cat(1,sample_valence,reconstructed_datasample');

        i=i+1; 
            
    end
    i=1;
end
%}
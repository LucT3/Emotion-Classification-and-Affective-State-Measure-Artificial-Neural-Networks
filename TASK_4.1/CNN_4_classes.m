%% TASK 4.1 (CNN for 4 classes)


%% Preparation of data
clear
close all
clc

%% Number of Classes to Classify
numberOfClasses = 4;

%% IMAGES LOADING

% Base image to get the proper size for the first layer of the CNN
base_img = imread('images/imgs_4_classes/anger/5.jpg');
base_img_size = size(base_img); %(224,224,3)

% loading images (300 filtered)
img_data = imageDatastore('images/imgs_4_classes', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% 70 per training, 30 per validation
[img_data_train, img_data_validation] = splitEachLabel(img_data, 0.7, 'randomized');



%% DATA AUGMENTATION

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

% Image augmentation for training and validation
augmented_image_data_train = augmentedImageDatastore(base_img_size, img_data_train, 'DataAugmentation', imageAugmenter);
augmented_image_data_validation = augmentedImageDatastore(base_img_size, img_data_validation);


%% CNN DEFINITION 

%CNN 4-class layers (anger,disgust,fear,happiness)
net_layers = [
    imageInputLayer(size(base_img)) %224,224,3
    
    convolution2dLayer(16,8,'Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer %tested also leakyReluLayer
     
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,16,'Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,32,'Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride' ,2)
    
    fullyConnectedLayer(80)
    fullyConnectedLayer(numberOfClasses)
    
    softmaxLayer
    
    classificationLayer
];


%% NETWORK TRAINING

%Training options definition
training_options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 40, ... %original 40
'MaxEpochs', 100, ... %original is 200
'Shuffle','every-epoch', ...
'ValidationData', img_data_validation, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

%show network
%analyzeNetwork(net_layers)

%network training
net = trainNetwork(augmented_image_data_train,net_layers,training_options);


%% RESULTS

%classification on validation set
YPred = classify(net,img_data_validation);
YValidation = img_data_validation.Labels;
accuracy_validation = sum(YPred == YValidation)/numel(YValidation)
plotconfusion(YValidation, YPred, 'validation 4 classes');

%Samples
idx = randperm(numel(img_data_validation.Files),12);
figure
for i = 1:12
    subplot(3,4,i)
    I = readimage(img_data_validation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
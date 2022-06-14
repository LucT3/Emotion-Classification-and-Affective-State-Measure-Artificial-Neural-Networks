%% TASK 4.1 (CNN for 2 classes- anger and happiness)


%% Preparation of data
clear
close all
clc

%% Number of Classes to Classify
numberOfClasses = 2;

%% IMAGES LOADING

% Base image to get the proper size for the first layer of the CNN
base_img = imread('images/imgs_2_classes/anger/5.jpg');
base_img_size = size(base_img); %(224,224,3)

% loading images (300 filtered)
img_data = imageDatastore('images/imgs_2_classes', ...
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

%CNN 2-class layers (anger and happiness emotions)
net_layers = [
    imageInputLayer(size(base_img)) %224,224,3
    
    convolution2dLayer(16,6,'Stride',4,'Padding','same')
    batchNormalizationLayer
    reluLayer 
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,12,'Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride' ,2)
    
    convolution2dLayer(4,24, 'Padding' , 'same' )
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride' ,2)
    
    fullyConnectedLayer(40)
    fullyConnectedLayer(numberOfClasses)
    
    softmaxLayer
    
    classificationLayer
];


%% NETWORK TRAINING

%Training options definition
training_options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 40, ...
'MaxEpochs', 100, ... %original is 200
'Shuffle','every-epoch', ...
'ValidationData', img_data_validation, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

%network training
net = trainNetwork(augmented_image_data_train,net_layers,training_options);


%% RESULTS

%classification on validation set
YPred = classify(net,img_data_validation);
YValidation = img_data_validation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy 2 classes is %8.2f%%\n',accuracy*100);
figure
plotconfusion(YValidation, YPred, 'validation 2 Classes');


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

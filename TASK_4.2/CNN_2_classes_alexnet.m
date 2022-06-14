%% CNN - 2 classes classification reusing alexnet

clc;
clear;
close all;


%% Constants and Parameters
numberOfClasses = 2;

%% IMAGES LOADING

% loading images (300 filtered) 
img_data = imageDatastore('images/imgs_2_classes', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% 70 per training, 30 per validation
[img_data_train, img_data_validation] = splitEachLabel(img_data, 0.7, 'randomized');


%% ALEXNET FINE-TUNE
net = alexnet
%analyzeNetwork(net)


%Replacing final Layers
original_layers = net.Layers(1:end-3);

%replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer
net_layers = [
    original_layers
    fullyConnectedLayer(numberOfClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%% NETWORK TRAINING 

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

% data augmentation for training and resizing for validation images (227x227x3)
input_size = net.Layers(1).InputSize
augmented_image_data_train = augmentedImageDatastore(input_size(1:2), img_data_train, 'DataAugmentation', imageAugmenter);
augmented_image_data_validation = augmentedImageDatastore(input_size(1:2), img_data_validation);



% Training options
training_options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',augmented_image_data_validation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% train the network
new_CNN = trainNetwork(augmented_image_data_train, net_layers, training_options);



%% NETWORK TESTING (RESULTS AND SAMPLES)

%classification on validation set
YPred = classify(new_CNN, augmented_image_data_validation);

YValidation = img_data_validation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy 2 classes is %8.2f%%\n',accuracy*100);
figure
plotconfusion(YValidation, YPred, 'validation 2 Classes');


%accuracy calculation, method 2
accuracy2 = mean(YPred == YValidation)


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

%% run_PCA_KNN

%Develop Prinicipal Component Analysis (PCA) method of
%dimensionality reduction and apply kNN method of classification.

clc;
clear all;
close all;

%% Load dataset
filename ='Sample_MNIST.mat';
data = load(filename);

%% Display 100 samples
m = size(data.X, 1);
cases = randperm(size(data.X, 1));
cases = cases(1:100);

%Display data
displayData(data.X(cases, :));
title('Figure 1. Input Data')

%% Training and Validation Datasets

[rows, columns] = size(data.X);
PickCases = 0.30 ;
idx = randperm(m);
load('indices.mat');
TrainingSet = data.X(train_idx,:) ;
ValidationSet= data.X(valid_idx,:) ; 

%% Train and test labels

TrainLabels = data.y(train_idx)';
ValidationLabels = data.y(valid_idx)';

%% PCA Transformation

N= 20;
[ATrain, YTrain, EigenValuesTrain] = PCA_transformation(TrainingSet, N);
 

%% NOTE : problem with plotconfusion running time, instead used confusionchart.

%%
YValidation=ValidationSet*ATrain;

%% Validation Results

ModelTrain = fitcknn(YTrain,TrainLabels,'NumNeighbors',10,'Standardize',1, 'Distance','euclidean', ...
  'DistanceWeight', 'squaredinverse');


%% Prediction

%for training set
YValidationPred = predict(ModelTrain,YValidation);

%% Plot confusion matrix
figure(2);
plotconfusion(ind2vec(ValidationLabels),ind2vec(YValidationPred'));


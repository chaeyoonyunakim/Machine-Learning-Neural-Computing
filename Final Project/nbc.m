clc; clear;
%% Load CIFAR-10 Dataset: pre-processed in python
load train_dataset_array_reshape.csv
load test_dataset_array_reshape.csv
load train_dataset_targets.csv
load test_dataset_targets.csv
%% Split train dataset into 70%Train + 30%Validation
rng default %to prevent any randomness for the randsample

n = size(train_dataset_array_reshape,1);                       
idxTrn = false(n,1);                
idxTrn(randsample(n,round(0.7*n))) = true; %filter index of the 70% for Train
idxVal = idxTrn == false; %take the opposite 30% for Validation

X_Train = train_dataset_array_reshape(idxTrn,:);
X_Validation = train_dataset_array_reshape(idxVal,:);
y_Train = train_dataset_targets(idxTrn,:);
y_Validation = train_dataset_targets(idxVal,:);
%% Check Categories and Class Size
tabulate(y_Train)
tabulate(y_Validation)
%% Training the Naive Bayes Classifier(NBC)
rng default
tic;
Mdl1 = fitcnb(train_dataset_array_reshape, train_dataset_targets);
toc;
%% Run the NBC Model on the Validation Dataset
Ynew_Val = predict(Mdl1, X_Validation);
%% Check the Result
accuracy = sum(Ynew_Val == y_Validation)/numel(y_Validation)
%% Visualize
T = array2table(y_Validation);
Tnew = array2table(Ynew_Val);
Tconcatenation = [T Tnew];

h = heatmap(Tconcatenation,'y_Validation','Ynew_Val');
h.Title = 'Naive Bayes Classification(NBC) on the Validation CIFAR-10 dataset';
h.XLabel = 'Target Label';
h.YLabel = 'NBC Prediction';
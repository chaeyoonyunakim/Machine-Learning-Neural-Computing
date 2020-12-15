clc; clear;
%% Load CatDog Sub-Dataset: pre-processed in python
load CatDog_train_reshape.csv
load CatDog_test_reshape.csv
load CatDog_train_targets.csv
load CatDog_test_targets.csv
%% Convert class 3(Cat) to 1, class 5(Dog) to 2
temp = CatDog_train_targets == 3;
CatDog_train_targets((temp)) = 1; %Cat
CatDog_train_targets((~temp)) = 2; %Dog
temp = CatDog_test_targets == 3;
CatDog_test_targets((temp)) = 1; %Cat
CatDog_test_targets((~temp)) = 2; %Dog
%% Split train dataset into 70%Train + 30%Validation
rng default %to prevent any randomness for the randsample

n = size(CatDog_train_reshape,1);                       
idxTrn = false(n,1);                
idxTrn(randsample(n,round(0.7*n))) = true; %filter index of the 70% for Train
idxVal = idxTrn == false; %take the opposite 30% for Validation

X_Train = CatDog_train_reshape(idxTrn,:);
X_Validation = CatDog_train_reshape(idxVal,:);
y_Train = CatDog_train_targets(idxTrn,:);
y_Validation = CatDog_train_targets(idxVal,:);
%% Check Categories and Class Size
tabulate(y_Train)
tabulate(y_Validation)
tabulate(CatDog_test_targets)
%% Training the Naive Bayes Classifier(NBC) for the baseline of comparison
Training_time_NBC =[];
tStart = tic;
Mdl1 = fitcnb(X_Train, y_Train);
tEnd = toc(tStart);
Training_time_NBC = [tEnd]; %store training time in a list
%% Run the NBC Model on the Train CatDog Sub-Dataset
Accuracy_Train_NBC = [];
Ynew_Train = predict(Mdl1, X_Train);
Result = sum(Ynew_Train == y_Train)/numel(y_Train);
Accuracy_Train_NBC = [Result]; %store training accuracy in a list
%% Run the NBC Model on the Validation CatDog Sub-Dataset
Accuracy_Val_NBC = [];
Ynew_Val = predict(Mdl1, X_Validation);
Result = sum(Ynew_Val == y_Validation)/numel(y_Validation);
Accuracy_Val_NBC = [Result]; %store validation accuracy in a list
%% Visualize the Classification Accuracy of the NBC Model Prediction
T = array2table(y_Validation);
Tnew_Val = array2table(Ynew_Val);
Tconcatenation = [T Tnew_Val];

%and confirm the ML Model works well on validation set prior to test-stage
h = heatmap(Tconcatenation,'y_Validation','Ynew_Val');
h.Title = 'Naive Bayes Classification(NBC) on the Validation Sub-dataset(CatDog)';
h.XLabel = 'Target Label';
h.YLabel = 'NBC Prediction';
%% Run the Initial NBC Model on the Test CatDog Sub-Dataset
Accuracy_Test_NBC = [];
Ynew_Test = predict(Mdl1, CatDog_test_reshape);
Result = sum(Ynew_Test == CatDog_test_targets)/numel(CatDog_test_targets);
Accuracy_Test_NBC = [Result]; %store test accuracy in a list
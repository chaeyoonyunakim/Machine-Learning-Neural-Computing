%% Encode numerical label to the categorical target to proceed the Logistic Regression Classifier(LRC)
y_Train_Categorical = categorical(y_Train);
y_Val_Categorical = categorical(y_Validation);
%% Training the Logistic Regression Classifier(LRC) for the baseline of comparison
Training_time_LRC =[];
tStart = tic;
[B,dev,stats] = mnrfit(X_Train, y_Train);
tEnd = toc(tStart)
Training_time_LRC = [tEnd]; %store training time in a list
%% Run the LRC Model on the Train CatDog Sub-Dataset
pred_Train = mnrval(B, X_Train);
[m, inx] = max(pred_Train, [], 2); %argmax to return the highest probability result
Accuracy_Train_LRC = [];
Result = sum(inx == y_Train)/numel(y_Train);
Accuracy_Train_LRC = [Result]; %store training accuracy in a list
%% Run the LRC Model on the Validation CatDog Sub-Dataset(Syntactic error)
pred_Val = mnrval(B, X_Validation);
[m, inx] = max(pred_Val, [], 2);
Accuracy_Val_LRC = [];
Result = sum(inx == y_Validation)/numel(y_Validation);
Accuracy_Val_LRC = [Result]; %store validation accuracy in a list
%% Visualize the Classification Accuracy of the LRC Model Prediction(Syntactic error)
T = array2table(y_Validation);
Tnew_Val = array2table(pred_Val);
Tconcatenation = [T Tnew_Val];

%and confirm the ML Model works well on validation set prior to test-stage
h = heatmap(Tconcatenation,'y_Validation','pred_Val');
h.Title = 'Logistic Regression Classifier(LRC) on the Validation Sub-dataset(CatDog)';
h.XLabel = 'Target Label';
h.YLabel = 'LRC Prediction';
%% Run the Initial LRC Model on the Test CatDog Sub-Dataset
Accuracy_Test_LRC = [];
pred_Test = mnrval(B, CatDog_test_reshape);
[m, inx] = max(pred_Test, [], 2);
Accuracy_Val_LRC = [];
Result = sum(inx == CatDog_test_targets)/numel(CatDog_test_targets);
Accuracy_Test_LRC = [Result]; %store test accuracy in a list
%% Apply the four PCA object k value on the LRC Model
%use same pca_obj = [30, 50, 80, 100] as nbcCatdogpca.m does
%Accuracy_Val_LRC =[]; %comment out if you have the baseline result from lrcCatdogbase.m
Accuracy_Val_pca_LRC =[]; %store accuracy result for later purpose choosing the best obj for the test
for i=1:length(pca_obj)
    [coeff,score,latent,~,explained] = pca(X_Train, 'NumComponents', i);
    pca_Train = X_Train*coeff; %project a new point to PCA new basis
    pca_Validation = X_Validation*coeff; %projection for the validation data
    tStart = tic;
    [B,dev,stats] = mnrfit(pca_Train, y_Train);
    tEnd = toc(tStart);
    Training_time_LRC = [Training_time_LRC, tEnd]; %append on the baseline
    
    pred = mnrval(B, pca_Validation);
    [m, inx] = max(pred, [], 2); %argmax to return the highest probability result
    Result = sum(inx == y_Validation)/numel(y_Validation);
    Accuracy_Val_LRC = [Accuracy_Val_LRC, Result]; %append on the baseline
    Accuracy_Val_pca_LRC = [Accuracy_Val_pca_LRC, Result]; %accuracies for pca loop
end
%% Plot LRC Performance
figure
b = bar(round(Training_time_LRC, 3)); %LRC Training time
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')

figure
b = bar(round(Accuracy_Val_LRC, 2)); %LRC accuracy for the Validation CatDog Sub-Dataset
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
grid on
%% find the best LRC with PCA
[best, inx] = max(Accuracy_Val_pca_LRC, [], 2); %argmax to return the highest accuracy result
best_pca = pca_obj(inx); %take the value using the index for the best pca
%% Test the best LRC with PCA
[coeff,score,latent,~,explained] = pca(X_Train, 'NumComponents', best_pca);
pca_Train = X_Train*coeff;
pca_Test = CatDog_test_reshape*coeff;

[B,dev,stats] = mnrfit(pca_Train, y_Train); %train the best NBC

%Accuracy_Test_LRC =[];%comment out if you have the baseline result from lrcCatdogbase.m
pred_Test = mnrval(B, pca_Test); %test
[m, inx] = max(pred_Test, [], 2);
Result = sum(inx == CatDog_test_targets)/numel(CatDog_test_targets);
Accuracy_Test_LRC = [Accuracy_Test_LRC, Result];
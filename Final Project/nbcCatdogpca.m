%% Apply PCA on the CatDog Sub-Dataset and Plot Cumulative Variance Graph
[coeff,score,latent,~,explained] = pca(X_Train, 'NumComponents', 150);
cum_variance = cumsum(latent/sum(latent));
% to select the reasonable k values > 70%
figure
plot(cum_variance, 'rx-')
xlim([0 100]) %zoom x-axis depends on the plot overview
grid on
title('Cumulative variance of principal components')
xlabel('Principle component number')
%% Apply the four PCA object k value on the NBC Model
pca_obj = [30, 50, 80, 100]; %from 80% of the cumulative variance
Accuracy_Val_pca_NBC =[]; %store accuracy result for later purpose choosing the best obj for the test
for i=1:length(pca_obj)
    [coeff,score,latent,~,explained] = pca(X_Train, 'NumComponents', i);
    pca_Train = X_Train*coeff; %project a new point to PCA new basis
    pca_Validation = X_Validation*coeff; %projection for the validation data
    tStart = tic;
    Mdl1 = fitcnb(pca_Train, y_Train);
    tEnd = toc(tStart);
    Training_time_NBC = [Training_time_NBC, tEnd]; %append on the baseline
    
    Ynew_Val = predict(Mdl1, pca_Validation);
    Result = sum(Ynew_Val == y_Validation)/numel(y_Validation);
    Accuracy_Val_NBC = [Accuracy_Val_NBC, Result]; %append on the baseline
    Accuracy_Val_pca_NBC = [Accuracy_Val_pca_NBC, Result]; %accuracies for pca loop
end
%% Plot NBC Performance
figure
b = bar(round(Training_time_NBC, 3)); %NBC Training time
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')

figure
b = bar(round((Accuracy_Val_NBC * 100), 2)); %NBC accuracy for the Validation CatDog Sub-Dataset
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center','VerticalAlignment','bottom')
grid on
%% Find the best NBC with PCA
[best, inx] = max(Accuracy_Val_pca_NBC, [], 2); %argmax to return the highest accuracy result
best_pca = pca_obj(inx); %take the value using the index for the best pca
%% Test the best NBC with PCA
[coeff,score,latent,~,explained] = pca(X_Train, 'NumComponents', best_pca);
pca_Train = X_Train*coeff;
pca_Test = CatDog_test_reshape*coeff;
tStart = tic;
Mdl1 = fitcnb(pca_Train, y_Train); %train the best NBC
tEnd = toc(tStart);

Ynew_Test = predict(Mdl1, pca_Test); %test
Result = sum(Ynew_Test == CatDog_test_targets)/numel(CatDog_test_targets);
Accuracy_Test_NBC = [Accuracy_Test_NBC, Result]
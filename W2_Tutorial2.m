%%Task1: Load hospital.mat dataset
load hospital

%Covariance(Weight, Blood pressure)
hospital.SysPressure = hospital.BloodPressure(:,1);
hospital.DiaPressure = hospital.BloodPressure(:,2);
% C1 = cov(hospital.Weight, hospital.SysPressure)
% C2 = cov(hospital.Weight, hospital.DiaPressure)

data=[hospital.Weight, hospital.BloodPressure];
C = cov(data)

% C = cov(hospital.Weight, hospital.BloodPressure)

%Coefficients(Age, Blood pressure)
R1 = corrcoef(hospital.Age, hospital.SysPressure)
R2 = corrcoef(hospital.Age, hospital.DiaPressure)
R = corrcoef(data)

%%Task2: Load fisheriris.mat dataset
load fisheriris

%Apply the Matlab functions fitctree
ctree = fitctree(meas,species); % create classification tree
view(ctree); % text description
view(ctree,'mode','graph') % graphic description

%Predict to the dataset
Xnew = [1:4]; %creat Xnew having the same number of columns as the original data X=meas
Ynew = predict(ctree,Xnew)

%Create a Cross-Validation Model
cvmodel = crossval(ctree);
L = kfoldLoss(cvmodel)
clear all; clc; close all;
%Machine Learning model output, target values
output = [0.1 0.3 0.6;0.2 0.6 0.2;0.3 0.4 0.3];
target = [0 0 1;0 1 0;1 0 0];

%Calculate:
Q1 = ClassE(output, target) %classification error=1/3
Q2 = MSE(output, target) %mean square error =(0.26+0.24+0.74)/3=0.413
Q3 = LSE(output, target) %least squares =(0.26+0.24+0.74)/2=0.62
Q4 = CE(output, target) %cross entropy=??? (given answer is 0.74)

function classError = ClassE(output, target) 
    [maxX, idx] = max(output,[],2);
    [maxT, ids] = max(target,[],2);
    out = idx ~= ids;
    classError = mean(out);
end

function MSE_func = MSE(output,target)
    B = sum(((output - target).^2)');
    MSE_func = mean(B);
end

function LSE_func = LSE(output,target)
    A = output - target;
    LSE_func = sum(sum(A.^2))./2;
end

function  CrossEntropy = CE(output,target)
    CrossEntropy = -1 .* sum(sum(target .* log(output)));
end
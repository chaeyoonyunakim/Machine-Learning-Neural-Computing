%Machine Learning model output, target values
output = [0.1 0.3 0.6;0.2 0.6 0.2;0.3 0.4 0.3]
target = [0 0 1;0 1 0;1 0 0]

%Calculate:
Q1 = ClassE(output, target)
Q2 = LSE(output, target)
Q3 = MSE(output, target)
Q4 = CE(output, target)

function classError = ClassE(output, target) 
    [maxX, idx] = max(output,[],2);
    [maxT, ids] = max(target,[],2);
    out = idx ~= ids;
    classError = mean(out);
end

function LSE_func = LSE(output,target)
    A = output - target;
    LSE_func = sum(sum(A.^2));
end

function MSE_func = MSE(output,target)
    B = sum(((output - target).^2)');
    MSE_func = mean(B);
end

function  CrossEntropy = CE(output,target)
    CrossEntropy = -1 .* sum(sum(target .* log(output)));
end
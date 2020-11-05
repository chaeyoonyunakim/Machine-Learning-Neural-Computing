%Training set:
XM = [6 180 12;
    5.92 190 11;
    5.58 170 11;
    5.92 165 10];
XF = [5 100 6;
    5.5 150 8;
    5.42 130 7;
    5.75 150 9];
YM = [1; 1; 1; 1];
YF = [0; 0; 0; 0];

%Caluate the mean and variance
mXM = mean(XM), mXF = mean(XF)
vXM = var(XM), vXF = var(XF)

%Probability P(male), P(female)
pM = 0.5;
pF = 1-pM;

%Test set:
sample1 = [6 130 8];
sample2 = [5.6 167 9];

pMs1 = pM*prod(normpdf(sample1, mXM, sqrt(vXM)))
pFs1 = pF*prod(normpdf(sample1, mXF, sqrt(vXF)))
pMs2 = pM*prod(normpdf(sample2, mXM, sqrt(vXM)))
pFs2 = pF*prod(normpdf(sample2, mXF, sqrt(vXF)))
%pMd = normpdf(sample1(:,1), mXM, sqrt(vXM))
%pFd = normpdf(sample2(:,1), mXF, sqrt(vXF))

Ynew = cat(1, YM, 1);

%Classification:
Q1 = ClassE(cat(1, XM, sample1), cat(1, YM, 1))

function classError = ClassE(output, target) 
    [maxX, idx] = max(output,[],2);
    [maxT, ids] = max(target,[],2);
    out = idx == ids;
    classError = mean(out);
end
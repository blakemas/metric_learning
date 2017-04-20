d = 5;
n= 2000;
sigma = eye(d);
X = mvnrnd(zeros(d,1), sigma, n);
stdstat = nuc_norm(X'*X)/n*norm(X*X'/n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma = zeros(5); 
sigma(1,1) = 4.9;
for i=2:d
    sigma(i,i) = .1/4;
end
X = mvnrnd(zeros(d,1), sigma, n);
biasedstat = nuc_norm(X'*X)/n*norm(X*X'/n)
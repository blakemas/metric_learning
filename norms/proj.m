p = [];
for n=2:50
x = zeros(n,1);
x(1) = sqrt(n);
A = normr(randn(1000,n));
p = [p; mean(abs(A*x))]
end
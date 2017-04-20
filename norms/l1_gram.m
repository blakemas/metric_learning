n = 500;

norms = [];
for d=2:floor(500)
    X = sqrt(d)*randl1(n, d);
    G = X*X';
    norms = [norms; norm_nuc(G/(d*n))];
end
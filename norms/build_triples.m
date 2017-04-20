function Y = build_triples(K, X, num_triplets)
n = size(K,2);
Y = [];
G = X'*K*X;
for t=1:num_triplets
    q = randsample(n,3)'; i=q(1); j=q(2); k=q(3);
    delta = G(k,k)-2*G(i,k)-(G(j,j)-2*G(i,j));
    % align triplet
    if delta < 0
        q = [q(1) q(3) q(2)];
        delta = -delta;
    end
    % flip according to noise model
     if rand() > 1/(1+exp(-delta))
          q = [q(1) q(3) q(2)];
     end
    Y = [Y; q];
end

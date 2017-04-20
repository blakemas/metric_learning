function [ p] = effective_rank(M )
    s = svd(M);
    p = sqrt(s(1)^2+s(2)^2)/norm(s);

end


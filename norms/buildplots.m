figure
ax1 = subplot(121)
plot(.2:.2:2, nuc_errs)
title('Prediction error')
xlabel('s - bound on nuclear norm is s*||M||_*')
ylabel('error on 100 random triplets')
subplot(122)
plot(.2:.2:2, nuc_ranks)
title('Proportion of energy in first two singular values')
xlabel('s - bound on nuclear norm is s*||M||_*')
ylabel('energy')


figure
ax1 = subplot(121)
plot(.2:.2:2, max_errs)
title('Prediction error')
xlabel('s - bound on max norm is s*||M||_\infty')
ylabel('error on 100 triplets')
subplot(122)
plot(.2:.2:2, max_ranks)
title('Proportion of energy in first two singular values')
xlabel('s - bound on max norm is s*||M||_\infty')
ylabel('energy')
import autograd.numpy as np
from autograd import grad
from autograd.core import primitive


@primitive
def absx(x):
    return np.sum(np.abs(x))


def make_grad_absx(ans, x):
    def gradient_product(g):
        s = np.sign(x)
        return np.full(x.shape, g)*s
    return gradient_product
absx.defgrad(make_grad_absx)


grad_absx = grad(absx)
x = np.array([[5., -2.],[3, 2.]])
print 'value', absx(x)
print 'gradient', grad_absx(x)

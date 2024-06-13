"""
test the nonlinear fixed point iteration and its sensitivity (gradient)
"""
import torch
from torch import nn

def loss_fn(z):
    return torch.sum(z**2, axis = -1).mean()

class FixedPointLayer(torch.nn.Module):
    def __init__(self, W, tol = 1e-4, max_iter = 1):
        super(FixedPointLayer, self).__init__()
        self.W = torch.nn.Parameter(W, requires_grad = True)
        self.tol = tol
        self.max_iter = max_iter
        # implement by vmap
        self.implicit_model = torch.vmap(self.implicit_model_)
        self.jac_batched = torch.vmap(torch.func.jacfwd(self.implicit_model_, argnums = 0))

    def implicit_model_(self, z, x):
        return z - torch.tanh(self.W @ z + x)
    
    def newton_step(self, z, x, g):
        J = self.jac_batched(z, x)
        z = z - torch.linalg.solve(J, g)
        return z, J

    def forward(self, x):
        self.iteration = 0
        with torch.no_grad():
            z = torch.tanh(x)
            while self.iteration < self.max_iter:
                g = self.implicit_model(z, x)
                self.err = torch.norm(g)

                if self.err < self.tol:
                    break

                # newton's method
                z, J = self.newton_step(z, x, g)
                self.iteration += 1
        
        # re-engage the autograd tape
        z = z - self.implicit_model(z, x)
        z.register_hook(lambda grad : torch.linalg.solve(J.transpose(1,2), grad))

        return z

def implicit_model(W, x, z):
    # the g function
    return z - torch.tanh(W @ z + x)


def implicit_model_test(W, x, z):

    if x.dim() == 1:
        # single sample case
        print('using the implicit model on one sample')
        z_ = z.clone().detach()
        x_ = x.clone().detach()
        
        dl_dz = torch.func.grad(loss_fn)(z_)
        df_dW, df_dz = torch.func.jacfwd(implicit_model, argnums = (0,2))(W, x_, z_)
        
        adjoint_variable = torch.linalg.solve(df_dz.T, -dl_dz)
        
        dl_dW = torch.einsum('i,ikl->kl', adjoint_variable, df_dW)
    
    else:
        print('using the implicit model on all samples')
        z = z.clone().detach()
        x = x.clone().detach()
        
        dl_dz = torch.func.grad(loss_fn)(z)

        jacfwd_batched = torch.vmap(torch.func.jacfwd(implicit_model, argnums = (0,2)), in_dims = (None, 0, 0))
        df_dW, df_dz = jacfwd_batched(W, x, z)

        adjoint_variable = torch.linalg.solve(df_dz.transpose(1,2), -dl_dz)

        dl_dW = torch.einsum('bi,bikl->kl', adjoint_variable, df_dW)
    
    print('dl_dz', dl_dz.shape)
    print('df_dW', df_dW.shape)
    print('df_dz', df_dz.shape)
    print('adjoint_variable', adjoint_variable.shape)
    print('dl_dW', dl_dW)


if __name__ == '__main__':

    torch.random.manual_seed(0)

    batch_size = 10
    n = 5
    W = torch.randn(n,n).double() * 0.5
    x = torch.randn(batch_size,n, requires_grad=True).double()

    print('using the model')
    model = FixedPointLayer(W, tol=1e-10, max_iter = 50).double()
    
    # check with the numerical gradient
    torch.autograd.gradcheck(model, x, check_undefined_grad=False, raise_exception=True)

    z = model(x)
    loss = loss_fn(z)
    loss.backward()
    print(model.W.grad)
    
    # implicit model method
    implicit_model_test(W, x[0], z[0])
    implicit_model_test(W, x, z)



    



    
    

    
"""
test the gradient calculated by the OptNet and the cvxpylayer

a implementation of the differentiable convex layer
forward pass: solve the optimization problem using the cvxpy sample by sample
backward pass: use the implicit function theorem to compute the gradient
"""

import torch
import torch.nn as nn
import cvxpy as cp
import torch.autograd as autograd
from itertools import accumulate
from cvxpylayers.torch import CvxpyLayer

class OptLayer(nn.Module):
    def __init__(self, variables, parameters, objective, 
                inequalities, equalities, **cvxpy_opts):
        super().__init__()
        self.variables = variables
        self.parameters = parameters

        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts
        
        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(
            cp.Minimize(objective(*variables, *parameters)), 
            self.cp_inequalities + self.cp_equalities
                                )
        
    def forward(self, *batch_params):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(batch_params[0].shape[0]):
            # compute a single sample in the minibatch and find its gradient.
            # more convenient method is to find the solution in parallel and find the gradient in parallel.

            # print('no of batch:', batch)
            # solve the optimization problem and extract solution + dual variables
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i,p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()
                self.problem.solve(**self.cvxpy_opts)
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_inequalities]
                nu = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_equalities]

            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z,lam,nu] for a in b])

            def mat(x):
                sz = [0] + list(accumulate([a.numel() for b in [z,lam,nu] for a in b]))
                val = [x[a:b] for a,b in zip(sz, sz[1:])]
                return ([val[i].view_as(z[i]) for i in range(len(z))],
                        [val[i+len(z)].view_as(lam[i]) for i in range(len(lam))],
                        [val[i+len(z)+len(lam)].view_as(nu[i]) for i in range(len(nu))])

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                # inside this is torch implementation
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                L = (self.objective(*z, *params) + 
                    sum((u*v).sum() for u,v in zip(lam,g)) + sum((u*v).sum() for u,v in zip(nu,dnu)))
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i]*g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            # compute residuals and re-engage autograd tape
            y = vec(z, lam, nu)
            y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))

            # compute jacobian and backward hook
            J.append(autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y))
            # y.register_hook(lambda grad,b=batch : torch.solve(grad[:,None], J[b].transpose(0,1))[0][:,0])
            # print('J shape:', J[-1].shape)
            y.register_hook(lambda grad, b = batch : torch.linalg.solve(J[b].transpose(0,1), grad))
            
            out.append(mat(y)[0])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        return out[0] if len(out) == 1 else tuple(out)     

class OptNet(nn.Module):

    def __init__(self, Psqrt_value, q_value, G_value, h_value, A_value):
        """
        an OptNet layer which takes the input b and return 
        """

        super().__init__()
        n = Psqrt_value.shape[1]
        m = G_value.shape[0]
        p = A_value.shape[0]

        # initialize the parameters
        self.Psqrt = nn.Parameter(Psqrt_value)
        self.q = nn.Parameter(q_value)
        self.G = nn.Parameter(G_value)
        self.h = nn.Parameter(h_value)
        self.A = nn.Parameter(A_value)

        # consider b as the output from the previous layer
        # the remaining parameters are the learnable parameters of the layer
        # create the OptLayer
        obj = lambda z, Psqrt, q, G, h, A, b: 0.5*cp.sum_squares(Psqrt @ z) + q@z if isinstance(z, cp.Variable) else 0.5*torch.sum((Psqrt @ z)**2) + q@z
        ineq = lambda z, Psqrt, q, G, h, A, b: G@z - h
        eq = lambda z, Psqrt, q, G, h, A, b: A@z - b

        self.layer = OptLayer(
            # allows abstract cvxpy variables and parameters
            # the sequence must be the same
            variables=[cp.Variable(n)],
            parameters=[cp.Parameter((n,n)), cp.Parameter(n), 
                        cp.Parameter((m,n)), cp.Parameter(m), 
                        cp.Parameter((p,n)), cp.Parameter(p)],
            objective=obj,
            inequalities=[ineq],
            equalities=[eq],
        )

    def forward(self, b):

        return self.layer(self.Psqrt.expand(b.shape[0], *self.Psqrt.shape), 
                            self.q.expand(b.shape[0], *self.q.shape), 
                            self.G.expand(b.shape[0], *self.G.shape), 
                            self.h.expand(b.shape[0], *self.h.shape), 
                            self.A.expand(b.shape[0], *self.A.shape), 
                            b)

def loss_fn(z):
    return z.square().sum()

def test_blog_example():

    print('blog example')
    n,m,p = 10,4,5
    
    # layer = formulate_optnet(n,m,p)

    z = cp.Variable(n)
    Psqrt = cp.Parameter((n,n))  # ensure the matrix is positive semidefinite
    q = cp.Parameter(n)
    G = cp.Parameter((m,n))
    h = cp.Parameter(m)
    A = cp.Parameter((p,n))
    b = cp.Parameter(p)

    def f_(z,Psqrt,q,G,h,A,b):
        # objective
        return 0.5*cp.sum_squares(Psqrt @ z) + q@z if isinstance(z, cp.Variable) else 0.5*torch.sum((Psqrt @ z)**2) + q@z
    def g_(z,Psqrt,q,G,h,A,b):
        # inequality
        return G@z - h
    def h_(z,Psqrt,q,G,h,A,b):
        # equality
        return A@z - b
    
    layer = OptLayer(variables=[z], parameters=[Psqrt,q,G,h,A,b], objective=f_, inequalities=[g_], equalities=[h_],
                    solver=cp.GUROBI) # the original paper uses OSQP, a ADMM-based QP solver

    
    torch_params = [torch.randn(2,*p.shape, dtype=torch.double).requires_grad_() for p in layer.parameters]
    autograd.gradcheck(lambda *x: layer(*x).sum(), tuple(torch_params), eps=1e-4, atol=1e-3, check_undefined_grad=False)

    print('all tests pass')

def test_optnet(Psqrt, q, G, h, A, b):
    # only consider single sample
    # into batched mode

    print('test_optnet')

    layer = OptNet(Psqrt, q, G, h, A)
    z = layer(b)
    loss = loss_fn(z)
    loss.backward()
    print('z:', z)
    print('loss:', loss)
    print('b.grad:', b.grad)

    return b.grad

def test_cvxpy(Psqrt, q, G, h, A, b):

    print('test_cvxpy')
    z = cp.Variable(Psqrt.shape[1])
    objective = cp.Minimize(0.5*cp.sum_squares(Psqrt @ z) + q@z)
    constraints = [G@z <= h, A@z == b]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    print('z:', z.value)

def test_cvxpylayer(Psqrt_value, q_value, G_value, h_value, A_value, b_value):

    # b_value is a batch of b
    print('test_cvxpylayer')
    z = cp.Variable(Psqrt_value.shape[1])
    Psqrt = cp.Parameter((Psqrt_value.shape[0], Psqrt_value.shape[1]))
    q = cp.Parameter(q_value.shape[0])
    G = cp.Parameter((G_value.shape[0], G_value.shape[1]))
    h = cp.Parameter(h_value.shape[0])
    A = cp.Parameter((A_value.shape[0], A_value.shape[1]))
    b = cp.Parameter(b_value.shape[1])

    prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(Psqrt @ z) + q@z),
                    [G@z <= h, A@z == b])
    
    layer = CvxpyLayer(prob, parameters = [Psqrt, q, G, h, A, b], variables = [z])

    z = layer(Psqrt_value, q_value, G_value, h_value, A_value, b_value)[0] # b_value is a batch of b
    
    loss = loss_fn(z)
    loss.backward()

    print('z:', z)
    print('loss:', loss)
    print('b.grad:', b_value.grad)    

    return b_value.grad


if __name__ == '__main__':
    """
    test the OptNet
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test the OptNet')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)

    # test_blog_example()

    """
    compare with the cvxpy layer
    """
    n, m, p = 10, 4, 5
    # layer = formulate_optnet(n, m, p)

    # parameters for initialization
    Psqrt = torch.randn(n, n, dtype=torch.double).requires_grad_()
    q = torch.randn(n, dtype=torch.double).requires_grad_()
    G = torch.randn(m, n, dtype=torch.double).requires_grad_()
    h = torch.randn(m, dtype=torch.double).requires_grad_()
    A = torch.randn(p, n, dtype=torch.double).requires_grad_()
    
    # parameters for the output of the previous layer
    batch_size = 2
    b = torch.randn(batch_size, p, dtype=torch.double).requires_grad_()
    # z = test_optnet(layer, Psqrt, q, G, h, A, b)

    b_grad_opt_net = test_optnet(Psqrt, q, G, h, A, b)
    
    # lets check with the cvxpy layer
    Psqrt = Psqrt.clone().detach().requires_grad_()
    q = q.clone().detach().requires_grad_()
    G = G.clone().detach().requires_grad_()
    h = h.clone().detach().requires_grad_()
    A = A.clone().detach().requires_grad_()
    b = b.clone().detach().requires_grad_()

    b_grad_cvxpylayer = test_cvxpylayer(Psqrt, q, G, h, A, b)

    print('=====================================')
    print('compare the gradients')
    print('=====================================')
    print('max diff:', torch.max(torch.abs(b_grad_opt_net - b_grad_cvxpylayer)))

    print('=====================================')
    # lets check with the cvxpy results
    for i in range(batch_size):
        test_cvxpy(Psqrt.detach().numpy(), q.detach().numpy(), 
                        G.detach().numpy(), h.detach().numpy(), 
                        A.detach().numpy(), b[i].detach().numpy())
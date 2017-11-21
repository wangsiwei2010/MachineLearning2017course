# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:18:56 2017

@author: sshss
"""

 -*- coding: utf-8 -*-
from cvxopt import matrix, log, div, spdiag, solvers
import pdb
from numpy.linalg import matrix_rank
import numpy as np
from kernels import *
import matplotlib.pyplot as plt

# SVM MKL
def conjugateRisk(gamma, Y, lambda_, risk = 'SVM'):
    # Input : matrix gamma
    # Output : value, gradient and hessian of R^{*}(-2*lambda*gamma) with respect to gamma
    n = gamma.size[0]
    if risk == 'SVM':
        val = -2*lambda_*sum([gamma[i]*Y[i] for i in range(n)])
        Df = -2*lambda_*matrix(Y).T
        H = matrix(0., (n,n))
    return val, Df, H

def domainConjugateRisk(Y, lambda_, risk = 'SVM'):
    # Constraints with a = -2*lambda_*gamma
    # Input : Y is a list
    n = len(Y)
    if risk == 'SVM':
        G = matrix(np.diag(2*lambda_*np.asarray([float(n)*Y[n_] for n_ in range(n)])))
        h = matrix([1. for i in range(n)])
    return G, h

def solveDualProblem(Kernel, Y, lambda_, risk ='SVM', optimal_value = 0):
    # Given a risk R, this solver gives the solution of the optimization problem :
    # max_gamma( - R^{*}(-2*lambda_*gamma) - lambda*gamma^{t}*K*gamma )
    # i.e min_gamma ( R^{*}(-2*lambda*gamma) + lambda*gamma^{t}*K*gamma )
    # For instance, in the case of SVM, this is the solution of :
    # max 2*lambda*sum_i(gamma) - lambda_ gamma^T K_eta gamma
    # under the constraints : y_i*gamma_i*lambda_*n >= -1
    # Warning : the cp solver is used for minimization. Be careful with the signs
    # and we suppose that R^{*}(lambda x) = lambda * R^{*}(x)
    # Inputs : - Kernel : array
    #          - Y : list
    #          _ lambda_ : float
    n = len(Y)
    Kernel = matrix(Kernel)
    def F(gamma = None, z = None):
        if gamma is None: return 0, matrix(0., (n,1))
        # val = - ( 2*lambda_*sum(gamma) - lambda_*gamma.T*Kernel*gamma )
        # Df = - ( 2*lambda_*matrix(1., (1,n)) - (2*Kernel*gamma).T )
        val1, Df1, H1 = conjugateRisk(gamma, Y, lambda_, risk)
        val = val1 + lambda_*gamma.T*Kernel*gamma
        Df = Df1 + lambda_*(2*Kernel*gamma).T 
        if z is None: return val, Df
        H =  z[0]*( lambda_*H1 + lambda_*2*Kernel )
        return val, Df, H

    G, h = domainConjugateRisk(Y, lambda_, risk)
    sol = solvers.cp(F, G=G, h=h)
    if optimal_value == 0:
        return sol['x']
    if optimal_value == 1:
        return -sol['primal objective'], sol['x']


def projector(y):
    # Projector on the closed convex : x in R^n, x >= 0, sum(x) = 1
    # Defined as an optimization problem
    # input : - y vector (array)
    # output : - projected vector as a CVX matrix
    solvers.options['maxiters'] = 200
    n = y.shape[0]
    y = matrix(y)
    def F(x = None, z = None):
        if x is None: return 0, matrix(1/float(n), (n,1))
        val = ((x-y).T*(x-y))[0]
        Df = 2*(x-y).T
        if z is None: return val, Df
        H = 2*z[0]*matrix(np.eye(n))
        return val, Df, H
    G = matrix(np.concatenate((np.array([[1 for i in range(n)],[-1 for i in range(n)]]), -np.eye(n)) , axis = 0))
    h = matrix(sum([[1., -1.],[0. for i in range(n)]], []))
    sol = solvers.cp(F,G,h)
    return sol['x']



def MKL(eta_init, Kernels, Y, lambda_, risk = 'SVM'):
    # We use firstSolverMKL to get the value of the functionnnal J function of the variable eta
    #  We then apply a projected gradient descent to minimize J
    #   Input : - Data Kernels : list of arrays 
    #           - eta_init : array
    #           - lambda_ : penalization parameter (float)
    #           - Y : list

    solvers.options['show_progress'] = True
    nbKernels = len(Kernels)
    eta = projector(eta_init)
    Kernel = np.zeros((Kernels[0].shape[0], Kernels[0].shape[1]))    
    for i in range(nbKernels): Kernel = Kernel + eta[i]*Kernels[i]    
    gamma = solveDualProblem(Kernel, Y, lambda_,risk)
    grad = np.array([-lambda_*(gamma.T*matrix(Kernels[i])*gamma)[0] for i in range(nbKernels)])
    k = 1
    while 1:
        alpha = 1/float(k)
        eta_ = projector(np.asarray(eta - alpha*matrix(grad)))
        k = k + 1
        print sum(abs(eta-eta_))/sum(abs(eta))
        if k > 100 or sum(abs(eta-eta_))/sum(abs(eta))  < 1e-2: break
        Kernel = np.zeros((Kernels[0].shape[0], Kernels[0].shape[1]))    
        for i in range(nbKernels): Kernel = Kernel + eta_[i]*Kernels[i]    
        gamma = solveDualProblem(Kernel, Y, lambda_, risk)
        grad = np.array([-lambda_*(gamma.T*matrix(Kernels[i])*gamma)[0] for i in range(nbKernels)])
        eta = eta_        
    return eta, gamma


def MKL_solver(Kernels, Y, lambda_, risk = 'SVM'):
    # Input : - Data array X : n * p
    #         - List array Y
    #         - lambda_ : regularization parameter
    nbKernels = len(Kernels)
    # eta_init = np.array([1/float(nbKernels) for i in range(nbKernels)])
    eta_init = np.array([1 for i in range(nbKernels)])*1/float(nbKernels)
    return MKL(eta_init, Kernels, Y, lambda_, risk)


def Classifier(X_train, X_test, kernel_functions, eta, gamma):
    # Evaluate binary classifier : Y in {-1, 1} from dual formulation
    # Input : - X_train array of size n*p
    #         - dual variable gamma
    f = 0
    n = X_train.shape[0]
    M = len(kernel_functions)
    #Loops through all kernel functions
    for m in range(M):
        kernel_func = kernel_functions[m]
        for j in range(n):
            f = f + eta[m]*gamma[j]*kernel_func(X_train[j,:], X_test)
    return f

def linearCombinationKernel(Kernels,eta):
    Kernel = np.zeros((Kernels[0].shape[0], Kernels[0].shape[1]))
    nbKernels = len(Kernels)
    for i in range(nbKernels):
        Kernel = Kernels[i]*eta[i]    
    return Kernel

if __name__ == '__main__':
    lambda_ = 1
    n = 100
    p = 10
    X = np.random.rand(n,p)
    n_train = int(0.75*n)
    X_train = X[:n_train]
    X_test = X[n_train:]
    Y = [1 if i % 2 == 0 else -1 for i in range(n)]
    Y_train = Y[:n_train]
    Y_test = Y[n_train:]
    nbKernelsPerType = 25 # perfect square for poly kernels list having same size as rbf kernels list
    Gamma = np.linspace(0.000001,0.1,nbKernelsPerType) 
    Gamma_poly = np.linspace(0.0001, nbKernelsPerType, np.sqrt(nbKernelsPerType)).tolist()
    Degree = np.linspace(0, 20, np.sqrt(nbKernelsPerType)).tolist()
    rbf_kernels = [create_rbf_kernel(g) for g in Gamma]
    poly_kernels = []
    for d in Degree:
        for g in Gamma_poly:
            poly_kernels.append(create_poly_kernel(d, g))
    kernel_functions = sum([[linear_kernel], rbf_kernels, poly_kernels], [])
    Kernels = get_all_kernels(X_train, kernel_functions)
    eta, gamma = MKL_solver(Kernels, Y_train, lambda_)
    resultsTest = [np.sign(Classifier(X_train, X_test[i,:], kernel_functions, eta, gamma)) for i in range(X_test.shape[0])]
    score_test = (1/float(len(Y_test)))*sum([1 for y, r in zip(Y_test, resultsTest) if y == np.sign(r)]) 
    pdb.set_trace()
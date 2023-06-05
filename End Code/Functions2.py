import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import expm
import sys
import time

#################################################
# Functions that are used by the other scripts
#################################################

def eight(num_points=100):
    '''Para metric function that creates a 8-figure in 2D .
    It outputs num_points many points in a tensor, where the coordinates are storred column-wise. 
    The first dimension is the point index.
    The second is are the rows of the point vector (is 2).
    The third is are the columns of the point vector (is 3)'''
    x=np.zeros((num_points,2,1))
    x_1=5*np.sin(2*np.linspace(0,2*math.pi,num_points))
    x_2=10*np.sin(np.linspace(0,2*math.pi,num_points)) 
        
    x[:,0,0]=x_1
    x[:,1,0]=x_2
    return x

def random_point_on_n_sphere(n):
    '''Selects a random point on a n dimensional hypersphere'''
    x = np.random.normal(size=(1,n,1))
    x = x / np.linalg.norm(x)
    return x

def mat_mult_points(R,x):
    '''Matrix multiplication of all the points in x by R.'''
    return np.einsum('ij,njk->nik', R, x)

# def mat_mult_basis(R,U):
#     return np.einsum('ij,njk->nik',R,U)

def loss(R,x,y):
    '''Loss function which is equal to the average 2-norm of Rx-y.'''
    all_norms=np.square(np.linalg.norm(mat_mult_points(R,x)-y, axis=1))
    #print('all_norms is ',all_norms)
    return all_norms.mean()

def multi_loss(y_it,y, l2_norm = False):
    '''Calculating the value of the loss function for multiple instances.
    Requires: y_it, as returned by Riem_grad_descent; y, desired point.
    Optional: l2_norm, is true if you want not the value of the loss function but the l2-norm between vectors.'''
    np.linalg.norm(y_it-y)
    num_it=np.shape(y_it)[0]
    losses=np.zeros(num_it)
    for it in range(num_it):
        if l2_norm:
            all_norms=np.linalg.norm(y_it[it]-y,axis=1)
        else:
            all_norms=np.square(np.linalg.norm(y_it[it]-y,axis=1))
        
        losses[it]=all_norms.mean() #taking the average
    
    return losses
    

def d_loss(R,x,y):
    '''Differential of the loss function f(R) (see report). 
    Warning: f(R) is not equal to the average l^2 norm.
    It is the average l^2 norm squared.'''
    num_points=x.shape[0]
    R_x_min_y=np.einsum('ij,njz->niz',R,x) - y
    d_l=(2/num_points)*np.einsum('niz,njz->ij',R_x_min_y,x)
    return d_l

def constr_skew_basis(dim=2):
    '''Constructs a basis of skew-symmetric matrices.
    Optional: dim, dimension of the embedding space of SO(n) i.e. n.
    Output: basis, the skew-symmetric basis S_{skew}(dim).'''
    num_basis=int(dim*(dim-1)/2)
    basis=np.zeros((num_basis,dim,dim))
    basis_it=0
    for i in range(0,dim-1,1):
        for j in range(i+1,dim,1):
            basis[basis_it,i,j]=1/2*np.sqrt(2)
            basis[basis_it,j,i]=-1/2*np.sqrt(2)
            basis_it=basis_it+1
   
    return basis



def calc_Riem_grad(Eucl_grad,R,U_U_trans):
    '''Calculating the Riemannian gradient by projecting the Eucl_grad onto the tangent space T_R M.
    Requires: Eucl_grad, the Euclidean gradient; R, the current iteration matrix;
    U_U_trans, the precomputed UU^T where U is a matrix contining the skew-symmetric basis vectors.
    Output: in_exp, the Riemannian gradient.'''
    
    dim=R.shape[0]
    
    R_transpose_Eucl_grad = np.einsum('ji, jk -> ik', R, Eucl_grad) # = R{-1}*Grad f(R)
    # uncomment if you want to add some pertubation
    #if np.all(Eucl_grad == np.transpose(Eucl_grad)):
        #print('Error warning: We are in the normal space of T_I M.' )
        # dim = np.shape(R)[0]
        # #adding small perturbation
        # pert = np.zeros((dim,dim))
        # pert[0,1] = 10**(-15)
        # pert[1,0] = -10**(-15)
        # R_transpose_Eucl_grad = R_transpose_Eucl_grad + pert

    R_transpose_Eucl_grad_vec = np.transpose(np.reshape(R_transpose_Eucl_grad,dim*dim,order='C'))
    #print('U_U_trans is: ', U_U_trans)
    #print('R_transpose_Eucl_grad_vec is: ', R_transpose_Eucl_grad_vec)
    in_exp = np.einsum('ij, j -> i' , U_U_trans, R_transpose_Eucl_grad_vec) #projecting on T_I G
    in_exp = np.reshape(in_exp, (dim, dim), order='C') #reshaping into matrix
    #print("argument in exponential is: ", in_exp)
    return in_exp #=grad f(R)

def update_R(R,x,y,U_U_trans,eta, info=False):
    '''One update step. R is the result from the previous iteration (or the initial guess).
    x are the point that should be rotated to y. U is a basis of the tangent space at identity T_I M.
    eta is the learning rate.'''
    #calculating Euclidean gradient
    Eucl_grad=d_loss(R,x,y)
    #uncommment if you want to check for symmetric Eucl_grad
    # if np.all(Eucl_grad == np.transpose(Eucl_grad)):

    #     print('Error warning: Euclidean gradient belongs to the normal space of T_I M.' )
    #     dim = np.shape(R)[0]
    #     #adding small perturbation
    #     pert = np.zeros((dim,dim))
    #     pert[0,1] = 10**(-15)
    #     pert[1,0] = -10**(-15)
    #     Eucl_grad = Eucl_grad + pert
    
    #print('Euclidean gradien is :', Eucl_grad)
    
    # retraction using QR decomposition
    #R_update, upper_tri = np.linalg.qr(R-eta*Riem_grad)
    
    #retraction using Riemanian retraction
    in_exp=calc_Riem_grad(Eucl_grad,R,U_U_trans)
    #print('in_exp is :', in_exp)
    R_exp = expm(-eta*in_exp)
    R_update = np.einsum('ij, jk-> ik', R, R_exp)
    
    
    if info:
        print("The new R is: ", R_update)
        print("The determinant is: ", np.linalg.det(R_update))
        print("Should be idenity: ", np.einsum("ij,kj -> ik", R_update, R_update))
        
    return R_update

def Riem_grad_descent(R_0,x,y,max_it,eta, info=False, add_init=True):
    '''Perform the Riemannian gradient descent algorithm.
    Requires: R_0, the initial guess; x, the initial image; y, the desired image;
    max_it, the maximum number of iterations; eta, the learning rate.
    Optional: info, boolean if you want extra information during iteration steps;
    add_init, boolean if True adds x to y_it at position zero.
    Output: R, final rotation matrix; y_it, all the images.'''
    #initialize end results
    n_points=x.shape[0]
    n = x.shape[1]
    y_it=np.zeros((max_it+add_init,n_points,n,1))
    
    if add_init:
        y_it[0,:,:,:]=mat_mult_points(R_0, x) #adds initial guess to y_it
    
    
    R=R_0
    
    #precompute U_U_trans
    U_U_trans=construct_U_U_trans(n)
    #print(U_U_trans)

    #iteration step
    for it in range(0,max_it):
        #print('IT number: ', it+1)
        #calculating Euclidean gradient
        R=update_R(R,x,y,U_U_trans,eta, info)
        
        #print('Determinant of R is ', np.linalg.det(R))
        #print('R after iteration ',it+1)
        #print(R)
        y_it[it+add_init,:,:,:]=mat_mult_points(R,x) #the initial guess is not in this
    
    return R, y_it

def multi_sim(R_0, x_0, max_it, n_sim, eta):
    '''Performs n_sim with a maximum of max_it iterations.
    Returns: all_losses, avarage loss at every iteration step;
    conv_point, the iteration when the losses is smaller than 10 times computer precision
    conv_point is equal to max_it when we did not converge.'''
    #eps = sys.float_info.epsilon
    all_losses = np.zeros((n_sim, max_it+1))
    conv_points = np.ones(n_sim)*max_it
    n = np.shape(R_0)[0]
    y_all = np.zeros((n_sim, 1, n, 1))
    x_length = np.linalg.norm(x_0)
    execution_time = 0
    
    for sim in range(0, n_sim):
        
        #choosing random point on the hyper sphere
        y = x_length*random_point_on_n_sphere(n)
        #storing this y value so that we can excess it later
        y_all[sim] = y
        
        start_time = time.time()
        
        #perform the algorithm using Riemannian gradient descend
        R, y_it = Riem_grad_descent(R_0,x_0,y,max_it,eta)
        
        end_time = time.time()
        execution_time = end_time - start_time + execution_time
        
        losses = multi_loss(y_it, y)
        all_losses[sim] = multi_loss(y_it, y)

        # looking when it did converge
        for it in range(0,max_it+1):
            if losses[it]<10**(-30):
                conv_points[sim] = it
                break
            
    execution_time_av = execution_time / (n_sim*max_it) #average execution time for every iteration
        
    return all_losses, conv_points, y_all, execution_time_av


def construct_U_U_trans(n):
    '''Calculate U_U_trans in n dimensions'''
    U=constr_skew_basis(dim=n)
    num_basis=U.shape[0]
    
    #pre-calculation
    U_matr_transpose = np.reshape(U, (num_basis, n*n), order='C')
    U_matr = np.transpose(U_matr_transpose)
    U_U_trans = np.einsum('ij, jk -> ik', U_matr, U_matr_transpose)
    
    return U_U_trans

#####################
# Plotting functions
#####################

def plot_figure(x,y=None, step=1, info = False):
    '''Given a four dimensional array x, this function plots the figure.
    First dimension of x contains all the pictures
    Second dimensions of x are the datapoints per picture
    Third dimension is the row of the data (x and y)
    Fourth dimension is equal to the columns of the data (always 1)'''
    if info:
        print('given data to plot is ', x)
    #clear figure
    num_pictures, num_points_per_picture, n_dim =x.shape[0:3]
    num_points_per_picture=x.shape[1]
    plot_data=np.zeros((n_dim, num_points_per_picture))
    label_format = r'$y^{{{it_step:.0f}}}$'
    
    for picture in range(num_pictures):
        if (picture%step) == 0:
            plot_data = np.array([x[picture,:,dim,0] for dim in range(0,n_dim)])
            # print('plot_data is: ', plot_data)
            if n_dim==2:
                plt.plot(plot_data[0,:],plot_data[1,:],label=label_format.format(it_step=picture))
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                plt.xlabel(r'$x$')
                plt.ylabel(r'$y$')
            elif n_dim==3:
                ax = plt.gca(projection='3d')
                ax.plot3D(plot_data[0,:], plot_data[1,:], plot_data[2,:], label=label_format.format(it_step=picture))
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
                ax.set_zlabel(r'$z$')
            else:
                print("Given data is not 2 or 3 dimensional, so no visualization is given.")
                return
            
    #print('plot_data is', plot_data)
    
    #plotting y
    if (y is not None) and (n_dim < 4):
        plot_data = np.array([y[:,dim,0] for dim in range(0,n_dim)])
        if n_dim==2:
            plt.plot(plot_data[0,:], plot_data[1,:], label=r'$\gamma$')
            #or a scatter plot
            #plt.scatter(plot_data[0,:], plot_data[1,:], color='red', alpha =0.2, label=r'$y$')
        elif n_dim==3:
            ax.plot3D(plot_data[0,:], plot_data[1,:], plot_data[2,:], label=r'$\gamma$')
        else:
            print("The desired data is not of dimension 2 or 3")
            
    #set axis limits here
    plt.xlim([-12,12])
    plt.ylim([-12,12])
    plt.show()
    plt.legend(loc = 'upper right')
    
    
def plot_loss(losses, log=True):
    '''This function plots the value of the average l^2 norms.
    Requires: losses, as given by multi_losses().
    Optional: log, is True if you want the y-axis on a log scale.'''
    num_it=losses.shape[0]-1 #minus one, since the inital loss is also given
    x=[it for it in range(num_it+1)]
    if log:
        plt.semilogy(x, losses)
        #plt.title('Loss on a log scale')
    else:
        plt.plot(x,losses)
        #plt.title('Loss')
    plt.title(r'Value of the loss function $f(R)$')
    plt.xlabel(r'iteration step $t$')
    plt.ylabel(r'$f(R_t)$')
    plt.xlim([0, num_it])
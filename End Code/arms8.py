from scipy.optimize import root_scalar
from scipy.integrate import quad
import numpy as np
import time
import matplotlib.pyplot as plt
import math

import arms6 as arms6
import Functions2 as func2

################################
# arms8
################################
# Fitting a robotic arm along a curve.
# Improvement w.r.t. arms6 (which had no arc length initialization)
# In arms8 we initialize the t parameter of the gamma curve so that the arc length of
# the curve between two consequtive t values is equal to the arms length.
# The code below is compatible with any gamma curve. Therefore we need to solve 
# the arc_lenght problem. This is done using the scipy package.
###############################################################################
# To obtain the same results as in the thesis, please uncomment 
# the eight (2d) curve and/or the coil (3D) curve in arms6
###############################################################################

def arc_length(t):
    '''Returns the arc lenght of d_gamma'''
    arc = np.linalg.norm(arms6.d_gamma(t))
    #print('arc length is: ', arc)
    return arc


def calc_arc_length(t_0,t_1):
    '''Calculates the arc length from t_0 to t_1'''
    result, error = quad(arc_length, t_0, t_1)
    # print('result is: ', result)
    return result

def initialize_t(arm_length, n_arms, info=False):
    '''Calculates the initial positions of t, using the arclength.
    Requires: arm_length, the length of one arm piece as float;
    n_arms, number of arms as integer;
    Optional: t_max, maximum time parameter of the curve as float;
    info, boolean is True then printing info about the convergence of the root finding alg.'''
    t_0 = np.zeros(n_arms+1)
    
    for arm_it in range(1, n_arms+1):
        
        f = lambda t: calc_arc_length(t_0[arm_it-1], t) - arm_length
        
        zero_info = root_scalar(f, method='bisect', bracket=[t_0[arm_it-1], t_0[arm_it-1] + 2*arm_length ])
        
        if info:
            print('Finding root for arm_it ', arm_it,':')
            print(zero_info)
            
        t_0[arm_it] = zero_info.root
    
    return t_0[1:]

def Riem_grad_descent_with_arc_length_init(R_0, x_0, eta, eta_t, max_init_it, max_it, info=False, n_inter = 0):
    '''Same as Riemannian_grad_descent_multi_arms in arms6, but now with arc_length initialization.'''
    arms_length = np.linalg.norm(x_0)
    n_arms = np.shape(R_0)[0] #arc_length initialization of the t_0 variables
    
    R_it = np.zeros((max_init_it + max_it + 1, n_arms, n, n))
    y_it=np.zeros((max_init_it + max_it + 1, n_arms + 1, n, 1))
    t_it = np.zeros((max_init_it + max_it + 1, n_arms))
    
    print('t is initialized according to arc_length')
    t_0=initialize_t(arms_length, n_arms)
    print('the initial t are: ', t_0)
    
    R_it[0:max_init_it+1], y_it[0:max_init_it+1], t_it[0:max_init_it+1] = arms6.Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_init_it, info = info, n_inter = n_inter, updating_t=False)
    
    print(max_init_it, ' initial iterations are performed.')
    print('R_0 fitted on fixed gamma curve positions. Now, we unfix t.')
    R_0 = R_it[max_init_it]
    t_0 = t_it[max_init_it]
    
    R_it[max_init_it:], y_it[max_init_it:], t_it[max_init_it:] = arms6.Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it, info =info, n_inter=n_inter)
    
    t_diff = t_it[max_init_it+max_it] - t_it[max_init_it]
    if np.any(t_diff<0):
        print('Error warning: arc length is shorter than the length of the arm')
        print(t_diff)
    
    print(max_it, ' iterations are performed with t unfixed.')
    print('In total we have done ', max_init_it+max_it, 'iterations.')
    print('Algorithm finished, returning R_it, y_it, t_it')
    
    return R_it, y_it, t_it


##################################################################
# main code below
##################################################################
  

# Setting simulation variables
n=2 #dimension, please adjust the functions gamma and d_gamma accordingly
n_arms = 18
R_0 = arms6.identity_Rs(n, n_arms)
#theta = 1/4 * np.pi
#R_0[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) 
x_0=np.zeros((n,1))
x_0[0,0]=0.5
arms_length = np.linalg.norm(x_0)

# t_0=initialize_t(arms_length,n_arms)  
# print('print t_0')

#eta=np.array([0.5, 0.5, 0.5, 0.5, 0.5])
eta=2
eta_t=0.1
max_init_it = 50
max_it = 150
n_inter = 9
n_to_plot=4

# # printing some information
# y=arms6.retrieve_axis_positions(R_0, x_0)
# print('n_arms is: ', n_arms)
# print('initial coordinate are ', y)
# print('initial t is: ', t_0)


#performing the algorithm
start_time = time.time()
R_it, y_it, t_it = Riem_grad_descent_with_arc_length_init(R_0, x_0, eta, eta_t, max_init_it, max_it, n_inter = n_inter)
execution_time = time.time()-start_time
print('The algorithm ran for: ', execution_time, ' seconds')
print('So ', execution_time/(max_init_it+max_it), ' seconds for each iteration')

#analyses of the results
#calculating losses
loss = arms6.calc_losses(y_it, t_it)

#plotting results 
#calculating Riem_grad_norms

# plotting arm
plt.close('all')
fig = plt.figure()
if n==2:
    ax = fig.add_subplot(111)
elif n==3:
    ax = fig.add_subplot(111, projection='3d')
gamma_curve = arms6.gamma(np.linspace(0,1.5,10000))
func2.plot_figure(y_it, gamma_curve, step=math.floor((max_init_it + max_it)/n_to_plot))
#plotting fitting points on curve in the same figure
if n==2:
    x_scatter = arms6.gamma(t_it[-1])[:, 0]
    y_scatter = arms6.gamma(t_it[-1])[:, 1]
    plt.scatter(x_scatter, y_scatter)
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2, 1.2])
    for arm_it in range(n_arms):
        ax.text(x_scatter[arm_it], y_scatter[arm_it], str(arm_it+1), ha='center', va='bottom')
elif n==3:
    x_scatter = arms6.gamma(t_it[-1])[:, 0]
    y_scatter = arms6.gamma(t_it[-1])[:, 1] 
    z_scatter = arms6.gamma(t_it[-1])[:, 2]
    ax.scatter(x_scatter,y_scatter,z_scatter)
    ax.set_xlim([-0.2,2.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_zlim([0,1.5])
    for arm_it in range(n_arms):
        ax.text(x_scatter[arm_it][0], y_scatter[arm_it][0], z_scatter[arm_it][0], str(arm_it+1), ha='center', va='bottom')




plt.title('Robotic arm with arc length initialization')


#plotting loss functions
plt.figure()
arms6.plot_loss(loss,plot_all_losses=False) #set plot_all_losses to true to plot losses for every arm piece
plt.axvline(x=max_init_it, color='red', linestyle='dotted')
plt.title('Loss with arc length initialization')


#plotting max_dist
# plt.figure()
# arms6.plot_max_dist(y_it, t_it)

#plotting Riem grad w.r.t. to R
# plt.figure()
# all_Riem_grad_R_norms, all_dt_norms = arms6.plot_Riem_grad_R_norms(y_it, t_it, R_it, x_0, n_inter=n_inter)

#change dimensions to work with 3

            
    
















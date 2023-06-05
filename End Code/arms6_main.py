import arms6 as arms6
import Functions2 as func2

import matplotlib.pyplot as plt
import numpy as np
import time




################################################################
# arms6_main
################################################################

###############################################################################
# This is execute the algorithm in arms6 (without arc_length initialization)
# and plots some results
# system parameters can easily be adjusted
# choose a curve in arms6 by only uncommenting the desired curve gamma and its derivative d_gamma
# slight adjustment to the plotting range of figure 1 could be made to get better figures.
###############################################################################
# To obtain the same results as in the thesis, please uncomment the 2D parabola curve in arms6
###############################################################################
    

# Setting simulation variables
n=2 #dimension, please adjust the functions gamma and d_gamma in arms6 accordingly
n_arms = 3
R_0 = arms6.identity_Rs(n, n_arms) 
# R_0[1]=-R_0[1] #for dubbelvouwen test case
#theta = 1/4 * np.pi
#R_0[0] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
x_0=np.zeros((n,1))
x_0[0,0]=0.5
x_0_length = np.linalg.norm(x_0)
#t_0 = np.linspace(0.5*x_0_length, 0.5*n_arms*x_0_length, n_arms) 
t_0 = np.array([0 for i in range(n_arms)])
# t_0 = np.array([0.4,0,0.4]) #for dubbelvouwen test case

#eta=np.array([0.5, 0.5, 0.5, 0.5, 0.5])
eta=2 #similare to eta=1 when x_0_length=1
eta_t=0.1
max_it = 400
n_inter = 9 #n_inter = number of integration points per link - 1
n_to_plot=4

# printing some information
y=arms6.retrieve_axis_positions(R_0, x_0)
print('n_arms is: ', n_arms)
print('initial coordinates are ', y)
print('initial t is: ', t_0)
print('eta is: ', eta)
print('eta_t is: ', eta_t)


#performing the algorithm
start_time = time.time()
R_it, y_it, t_it = arms6.Riemannian_grad_descent_multi_arms(R_0, x_0, t_0, eta, eta_t, max_it, info=False, n_inter = n_inter)
execution_time = time.time()-start_time
print('The algorithm ran for: ', execution_time, ' seconds')
print('Plotting results ...')
#analyses of the results
#calculating losses
loss = arms6.calc_losses(y_it, t_it)

#plotting results 
#calculating Riem_grad_norms

# plotting arm and gamma curve
plt.close('all')
# plt.figure()
gamma_curve = arms6.gamma(np.linspace(0,2,10000))
func2.plot_figure(y_it, gamma_curve, step=max_it/n_to_plot)
# plt.xlim([-0.2,1])
# plt.ylim([-0.05,1])
plt.ylim([-1.5, 1.5])
plt.xlim([-1.2, 1.5])
plt.title('Robotic arm in 2D')

#plotting fitting points on curve in the same figure
x_scatter = arms6.gamma(t_it[-1])[:, 0]
y_scatter = arms6.gamma(t_it[-1])[:, 1]
plt.scatter(x_scatter, y_scatter)

for arm_it in range(n_arms):
    plt.text(x_scatter[arm_it], y_scatter[arm_it], str(arm_it+1), ha='center', va='bottom')

plt.title('Robotic arm without arc length initialization')

#plotting loss functions
plt.figure()
arms6.plot_loss(loss, plot_all_losses=False)
plt.xlim([0,max_it])
plt.title('Loss without arc length initialization')

# plotting max_dist
# plt.figure()
# arms6.plot_max_dist(y_it, t_it)

# plotting Riem grad w.r.t. to R
plt.figure()
all_Riem_grad_R_norms, all_dt_norms = arms6.plot_Riem_grad_R_norms(y_it, t_it, R_it, x_0, n_inter=n_inter)
plt.xlim([0,max_it])

print('plotting finished')

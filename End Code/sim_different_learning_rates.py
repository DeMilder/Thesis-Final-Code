import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt


#############################################
#Rotation in different dimensions comparison
#############################################

########################################################################
# added possibilty to save the results (see next line)
# these results are loaded when running plotting_learning_rates_sim.py
########################################################################

save_results = True #is True if you want the results to be saved as a npz file.

#iteration parameters
max_it=10

#number of simulation per dimension
n_sim = 10000
#dimension
n=5 #smaller dimensions gives maybe weird convergence rates
#lenght of the vectors 
x_length=1

x_0=np.zeros((1,n,1)) #the first dimension leaves room for multiple vectors
x_0[0,0,0]=x_length

#initial guess
R_0=np.identity(n)

#learning rates
eta = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
#eta = np.array([1,2,3,4,5]) #more optimal if x_length = 0.5

all_losses = np.zeros((eta.shape[0], n_sim, max_it+1))
 
#initializing random point. The same n_sim number of random points are selected for each eta.
y_all = np.zeros((n_sim, n, 1))
for sim in range(0,n_sim):
    y_all[sim] = func2.random_point_on_n_sphere(n)


for eta_it in range(0, len(eta)):
    print('eta is: ', eta[eta_it])
    for sim in range(0, n_sim):
        
        #perform the algorithm using Riemannian gradient descend
        R, y_it = func2.Riem_grad_descent(R_0,x_0,y_all[sim],max_it,eta[eta_it])

        all_losses[eta_it, sim] = func2.multi_loss(y_it, y_all[sim]) 

all_losses_med = np.median(all_losses, axis=1)

if save_results:
    np.savez('results_learning_rates_all_eta.npz', eta=eta, all_losses = all_losses, n=n, y_all = y_all)



# plt.close('all')
# #print(all_losses)
# plt.figure()
# x=np.array([k for k in range(0,max_it+1)])
# for k in range(0, eta.shape[0]):
#     plt.semilogy(x,all_losses_med[k,:], label=r'$\eta = {}$'.format(eta[k]))
#     loss_max = np.max(all_losses[k],axis=0)
#     loss_min = np.min(all_losses[k],axis=0)
#     plt.fill_between(x, loss_min, loss_max, alpha=0.2)


# plt.legend(loc = 'upper right')
# title = r'Convergence for various learning rates $\eta$ in $\mathbb{{R}}^{n}$'.format(n=n)
# plt.title(title)
    
# plt.xlabel('iteration step')
# plt.ylabel(r'$f(R_i)$')  
# plt.xlim([0,40])
# plt.ylim([10**(-16),10])


#plt.ylim([0.000001, 10])

# num_it=losses.shape[0]
# x=[it for it in range(num_it)]
# if log:
#     plt.semilogy(x, losses)
#     plt.title('Loss on a log scale')
# else:
#     plt.plot(x,losses)
#     plt.title('Loss')

# plt.xlabel('iteration step')
# plt.ylabel('Average 2-norm')     
        





import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
import time


##########################
#Rotation in different dimensions comparison
##########################

def bar_chart(all_losses, dim):
    ''' Creates an histogram of the iterations step of convergence.
    Requires: all_losses, every loss value for every dimension, simulation and iteration;
    dims, the tested dimensions.'''
    max_it = np.shape(all_losses)[2]
    n_sim = np.shape(all_losses)[1]
    n_dim = np.shape(all_losses)[0]
    
    #obtaining the frequencies   
    conv_hist = np.zeros((dim.shape[0], max_it))            
    for k in range(0, n_dim):
        for sim in range(0, n_sim):
            for it in range(0,max_it):
                if all_losses[k, sim , it] < 10**(-15):
                    conv_hist[k, it] = conv_hist[k, it] +1
                    break
    conv_hist = conv_hist/n_sim

    
    fig, ax = plt.subplots()
    index = np.arange(max_it)
    bar_width = 0.8/n_dim
    opacity = 0.8
    for k in range(0, n_dim):
        plt.bar(index+k*bar_width, conv_hist[k], bar_width,
        alpha=opacity,
        label=dim[k])
    
    plt.xlabel('number of iteration steps untill convergence')
    plt.ylabel('relative frequency')
    plt.title('Convergence histogram')
    plt.xticks(index + bar_width, np.arange(max_it))
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

#iteration parameters
max_it=10
eta=1 #learning rate

#number of simulation per dimension
n_sim = 10000
#plot_all_sim = True #set to true if you want to plot every single simulation
plot_bar = True
plot_min_max = True

#dimensions
dim = np.array([2, 3, 5, 10 , 20])
n_dim = len(dim)
labels = [r'$n = {}$'.format(dim[k]) for k in range(0,n_dim)]


all_losses = np.zeros((dim.shape[0], n_sim, max_it+1))
conv_points = np.zeros((n_dim, n_sim))
execution_time_av = np.zeros(n_dim)


for k in range(0, len(dim)):
    #starting vector
    n=dim[k] # the number of dimensions
    x_0=np.ones((1,n,1)) #the first dimension leaves room for multiple vectors
    x_0 = x_0 /np.linalg.norm(x_0)
    # x_0 = np.zeros((1,n,1))
    # x_0[0,0,0]=1

    #initial guess
    R_0=np.identity(n)
    
    #doing n_sim simulations for every dimension
    all_losses[k], conv_points[k], y_all, execution_time_av[k] = func2.multi_sim(R_0, x_0, max_it, n_sim, eta)


print('execution times: ', execution_time_av)

#plotting median
plt.close('all')
plt.figure()
x=np.array([k for k in range(0,max_it+1)])

all_losses_med = np.median(all_losses, axis=1)
all_losses_max = np.max(all_losses, axis=1)
all_losses_min = np.min(all_losses, axis=1)
colors = ['' for k in range(0,n_dim)]

for k in range(0, n_dim):
    line, = plt.semilogy(x,all_losses_med[k,:], label=r'$n = {}$'.format(dim[k]))
    colors[k] = line.get_color()
    if plot_min_max:
        plt.fill_between(x, all_losses_min[k], all_losses_max[k], alpha=0.2)
        
    
plt.legend(loc = 'upper right')
plt.title('Convergence in various dimensions (median)')
    
plt.xlabel(r'iteration step $t$')
plt.ylabel(r'$f(R_t)$')   
plt.xlim([0,max_it])

#plotting average
plt.figure()
x=np.array([k for k in range(0,max_it+1)])

for k in range(0, dim.shape[0]):
    plt.semilogy(x,np.mean(all_losses[k], axis=0), label=r'$n = {}$'.format(dim[k]))

plt.legend(dim, loc = 'upper right')
plt.title('Convergence in various dimensions (average)')
plt.legend()
    
plt.xlabel(r'iteration step $t$')
plt.ylabel(r'$f(R_t)$')   
plt.xlim([0,max_it])


#histogram plotting
if plot_bar:
    bar_chart(all_losses[0:3], dim)
    
plt.legend(labels)
    
# #plot multiple
# if plot_all_sim:
#     plt.figure()
#     for sim in range(0, n_sim):
#         plt.semilogy(x, all_losses[0, sim], label='sim '+ str(sim+1)) #only plot all the sim for the first dimension
        
#     plt.legend(dim, loc = 'upper right')
#     plt.title(r'Convergence of $f(R)$ for different $y$')
#     plt.legend(np.arange(1,n_sim))
#     plt.xlabel(r'iteration step $i$')
#     plt.ylabel(r'$f(R_i)$')   
#     plt.xlim([0,max_it])
#     plt.legend()
    
    
      


  
        




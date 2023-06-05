import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt

######################################################################
#plotting the results from the different learning rates simulations
######################################################################

plot_min_max = True
zoomed = True

#loading in the results
results = np.load('results_learning_rates.npz')
eta = results['eta']
all_losses = results['all_losses']
n = results['n']
n_eta, n_sim, max_it = np.shape(all_losses) #also contains the 0'th iteration, so max_it is one less
max_it = max_it - 1

all_losses_med = np.median(all_losses, axis=1)
all_losses_max = np.max(all_losses, axis=1)
all_losses_min = np.min(all_losses, axis=1)

#plotting the results
plt.close('all')
#print(all_losses)
plt.figure()
x=np.array([k for k in range(0,max_it+1)])
colors = ['w' for i in range(0,n_eta)]

for k in range(0, n_eta):
    line, = plt.semilogy(x,all_losses_med[k,:], label=r'$\eta = {}$'.format(eta[k]))
    colors[k] = line.get_color()
    if plot_min_max:
        plt.fill_between(x, all_losses_min[k], all_losses_max[k], alpha=0.2)

plt.legend(loc = 'upper right')
title = r'Convergence for various learning rates $\eta$ in $\mathbb{{R}}^{n}$'.format(n=n)
plt.title(title)
    
plt.xlabel(r'iteration step $i$')
plt.ylabel(r'$f(R_i)$')  
plt.xlim([0,60])
plt.ylim([10**(-32),10])

#######
#zoomed in figure
#######


if zoomed:
    plt.figure()
    for k in (3,4,5):
        plt.semilogy(x,all_losses_med[k,:], label=r'$\eta = {}$'.format(eta[k]), color = colors[k])
        if plot_min_max:
            plt.fill_between(x, all_losses_min[k], all_losses_max[k], alpha=0.2, color = colors[k])
    
    plt.xlim([0,5])
    plt.ylim([10**(-6),10])
    
    plt.xlabel(r'iteration step $i$')
    plt.ylabel(r'$f(R_i)$')  
    title = r'Convergence for various learning rates $\eta$ in $\mathbb{{R}}^{n}$'.format(n=n)
    plt.title(title)
    plt.legend()
    #plt.legend(loc='lower left')
    
#########################
#difference figure
################
    
plt.figure()
    
one_index = np.where(eta==1)[0][0]
#colors = ['b', 'g', 'orange', 'r', 'gray', 'c', 'deeppink', 'm', 'y', 'k', 'w']
diff_all =np.zeros((n_eta, n_sim, max_it+1))
for eta_it in range(0,n_eta):
    diff_all[eta_it] = all_losses[one_index]-all_losses[eta_it]
 
eta_select = np.arange(0, n_eta)
#eta_select = [1,2,3]
it = 2
print('Plotting histogram for eta=', eta[eta_select])
binwidth=0.1
diff_to_plot = [diff_all[k,:,it] for k in eta_select]

flier_props = dict(marker='o', markerfacecolor='red', markersize=0.1)
#whis=[0,100] as an argument lets all the datapoint be in the boxplot
plt.boxplot(diff_to_plot, vert=True, patch_artist=True, whis=[0,100], labels = eta[eta_select], flierprops=flier_props)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$f(R_{i}, \eta=1, y_k)-f(R_{i},\eta, y_k)$'.format(i=it))
plt.title(r'Iteration step {}'.format(it))
if it ==2:
    plt.ylim([-0.4,0.4])
elif it==1:
    plt.ylim([-0.5, 1.4])

# plt.hist( diff_to_plot, bins= np.arange(-1,1,binwidth), label=eta[k_select])
# plt.xlim([-1,1])
# plt.legend()
# for k in range(0,n_eta):
#     diff = all_losses[one_index]-all_losses[k]
#     diff_mean = np.mean(diff, axis=0)
#     plt.plot(x, diff_mean, label=r'$\eta = {}$'.format(eta[k]), color = colors[k])
    
    
# plt.legend()
# plt.xlim([1,4])
# plt.xticks((1,2,3,4))
# plt.ylim([-0.2,0.4])
# plt.xlabel(r'iteration step $i$')
# plt.ylabel(r'$f(R_i, \eta=1) - f(R_i, \eta)$')
# plt.title(r'Mean difference between $f(R_i, \eta)$ and $f(R_i, \eta=1)$')

# plt.ylim([-5,5])




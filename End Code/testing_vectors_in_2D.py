import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
import time


##########################
# Testing different rotation vectors in 2D
##########################

#iteration parameters
max_it=12
eta=1 #learning rate

#dimensions
n=2
n_theta = 10001

x_0 = np.zeros((1,n,1)) #the first dimension leaves room for multiple vectors
x_0[0,0,0] = 1
# x_0 = np.zeros((1,n,1))
# x_0[0,0,0]=1

#initial guess
R_0=np.identity(n)
theta = np.linspace(-np.pi, np.pi, num = n_theta)
conv_point = max_it*np.ones(n_theta)
losses = np.zeros((n_theta, max_it+1))

for theta_it in range(0, len(theta)):
    
    y = np.array([[[np.cos(theta[theta_it])],[np.sin(theta[theta_it])]]])
    
    R, y_it = func2.Riem_grad_descent(R_0,x_0,y,max_it,eta)
    
    losses[theta_it] = func2.multi_loss(y_it, y)
    
    for it in range(0,max_it+1):
        if losses[theta_it, it]<10**(-30):
            conv_point[theta_it] = it
            break
        
plt.close('all')
plt.figure()
plt.plot(theta, conv_point)
plt.xlabel(r'$\theta$')
plt.ylabel('number of iterations untill convergence')
plt.ylim([0,max_it])
plt.xlim([-np.pi, np.pi])
plt.title(r'Convergence rate for different intances of $y$')

theta_is_zero_index = int(np.median(np.arange(0,n_theta)))
theta_is_half_index = int(np.median(np.arange(theta_is_zero_index, n_theta)))
theta_is_kwart_index = int(np.median(np.arange(theta_is_zero_index, theta_is_half_index+1)))
theta_is_driekwart_index = int(np.median(np.arange(theta_is_half_index, n_theta)))

losses_to_plot = losses[[theta_is_zero_index, theta_is_kwart_index, theta_is_half_index, theta_is_driekwart_index, (n_theta-1)],:]

plt.figure()
plt.semilogy(np.arange(0, max_it+1), losses_to_plot[0], label= r'$\theta = 0$')  
plt.semilogy(np.arange(0, max_it+1), losses_to_plot[1], label= r'$\theta = \frac{1}{4} \pi$') 
plt.semilogy(np.arange(0, max_it+1), losses_to_plot[2], label= r'$\theta = \frac{1}{2} \pi$') 
plt.semilogy(np.arange(0, max_it+1), losses_to_plot[3], label= r'$\theta = \frac{3}{4} \pi$')
plt.semilogy(np.arange(0, max_it+1), losses_to_plot[4], label= r'$\theta = \pi$')  

plt.xlabel(r'iteration step $i$')
plt.xlim([-0.01,max_it])
plt.ylabel(r'$f(R_i)$')
plt.title(r'Convergence of the loss function $f(R)$ for different values of $\theta$')
plt.legend(loc = 'upper right')
      


  
        




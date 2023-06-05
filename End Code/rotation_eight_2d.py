import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
#scipy could also be useful

################################################
#Rotation of the eight figure with perturbations
################################################

#iteration parameters
max_it=10
eta=0.01#learning rate

#given is the eight figure
n_points=10000
x=func2.eight(n_points)
angle=0.5*np.pi

n_to_plot = 5

#simpel test case
#x=np.array([[[0],[0]],[[1],[0]]]) test case
#angle=0.5*np.pi

#the disired solutions
R_real=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle), np.cos(angle)]])
y=func2.mat_mult_points(R_real, x)

#adding small perturbation
mean = (0,0)
sigma=1
cov = sigma*np.eye(2)
pert = np.random.multivariate_normal(mean, cov, (n_points,1))
print(pert)
pert = pert[:,0,:, np.newaxis]
y_pert = y + pert
print('y is: ', y)
print('pert is: ', pert)
print('y_pert is: ', y_pert)

#initial guess
R_0=np.identity(2)

#perform the algorithm using Riemannian gradient descend
R, y_it = func2.Riem_grad_descent(R_0,x,y_pert,max_it,eta, info=True)

print('y_it_shape is: ', y_it.shape)
    
#plots iteration figures
plt.close('all') #close old figure windows (if there are any)
plt.figure()
func2.plot_figure(y_it,y,step=np.floor(max_it/n_to_plot))

#plotting the perterbed
plt.scatter(y_pert[:,0,:], y_pert[:,1,:], color='red', alpha =0.1, label=r'$\widehat{y}$')
plt.legend(loc='upper right')
plt.title('Rotations in 2D on perturbed data')


#plot loss function
plt.figure()
losses=func2.multi_loss(y_it,y_pert,l2_norm=True)
func2.plot_loss(losses)
plt.title(r'Average $2$-norm between $Rx^i$ and $\widehat{y^i}$')
plt.ylabel(r'$f_{t}$')






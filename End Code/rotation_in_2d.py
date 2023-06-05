import Functions2 as func2
import numpy as np
import matplotlib.pyplot as plt
import time

##########################
#Rotation of a vector in n-dimensions
##########################

#iteration parameters
max_it=8
eta=1 #learning rate
n_to_plot=max_it
#dimensions
n=2

#starting vector
x=np.ones((1,n,1))
x= x / np.linalg.norm(x) #the first dimension leaves room for multiple vectors

#the disired solutions
y = func2.random_point_on_n_sphere(n) #random point
#y = np.zeros((1,n,1))
y = -x
# y[0, -1 ,0] = 0.5*np.sqrt(2)
# y[0, -2, 0] = 0.5*np.sqrt(2)

#initial guess
R_0=np.identity(n)

#interesting choices:
x=np.zeros((1,2,1))
x[0,0,0]=1
theta = 1/4*np.pi
y=np.array([[[-1/2*np.sqrt(2)],[1/2*np.sqrt(2)]]])
y_with_zero = np.zeros((2,n,1))
y_with_zero[1:,:,:]=y
#R_0 = np.array([[np.cos(theta), - np.sin(theta)],[np.sin(theta),np.cos(theta)]] )

print('x is: ', x)
print('y is: ', y)
print('R_0 is: ', R_0)
print('eta is: ', eta)

y_it_with_zero = np.zeros((max_it+1,2,n,1))
start_time = time.time()
#perform the algorithm using Riemannian gradient descend
R, y_it = func2.Riem_grad_descent(R_0,x,y,max_it,eta, info=False)
y_it_with_zero[:,1:,:,:]=y_it
execution_time = time.time()-start_time

print("The code ran for: %s seconds" % execution_time)

is_id= np.einsum('ij,kj->ik',R,R)
    

#plot loss function
plt.close('all')
if n==2:
    plt.figure()
    func2.plot_figure(y_it_with_zero, y_with_zero, step=np.floor(max_it/n_to_plot))
    plt.title('Rotation in 2D')
    plt.xlim([-1.2, 1.2])
    plt.ylim([-0.2, 1.2])

plt.figure()
losses=func2.multi_loss(y_it,y)
func2.plot_loss(losses)


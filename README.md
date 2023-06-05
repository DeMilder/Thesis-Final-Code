# Thesis-Final-Code
In the End Code map you kan find code that I used to obtain the results presented in my thesis report.
Every file contains a lot of documentation and comments, so please open the files and read the comments. Some overall remarks are given below:

Functions2.py, arms6.py and arm8.py contain the main algorithms. 
Functions2.py contain the basic Riemannian gradient descent (RGD) algorithm to rotate one vector.
arms6.py contains the RGD algorithm for robotic arms, which does NOT use arc lenght initialization
arms8.py contains the RGD algorithm for robotic arms, which does use arc lenght initialization

Then we have some other files which perform the alogirithm and/or plot some results
\begin{itemize}
\item rotation_in_2d.py is a simple script to rotate a vector in 2d. This one is friendly to get an understand of the basic code
\item testing_vectors_in_2d.py is a script which test the performance of the vector-rotation algorithm in 2d for different instances
\item rotation_eigh_2d.py is a script which rotate a eight figure to rotated but perturbed version of itself
\item sim_different_dimensions.py tests the basic RGD algorithm in different dimensions
\item sim_different_learning_rates.py tests the basic RGD algorithm for different learning rates, this outputs a file which can be read by plotting_learning_rate_sim.py
\item plotting_learning_rate_sim.py plots the results obtained by running sim_different_learning_rates.py
\item arms6_main.py executes the RGD for the robotic arm (without arc length initialization)
\end{itemize}


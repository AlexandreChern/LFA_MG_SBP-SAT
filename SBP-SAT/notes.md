## On the $\omega$ values for the SBP-SAT methods


The test has shown that omega values need to be larger than 1 for the SOR method to get accelerated convergence. Ideal $omega$ values should be around 1.9. The residual plots show unsmoothed error distribution.


Notes from April 6
Using more iterations on the for smoothers would accelerate the MG performance significantly. Once steps of smoothers are enough to smooth results on different levels. The performance of the SBP-SAT MG matches the performance of the central finite difference with inject with the same MG configurations.


The bad performance of MG for the SBP-SAT method is very likely coming from the smoothers. The Jacobi / Gauss-Seidel method is not a good smoother on the coarsest grid. While they perform well with the central finite difference with injected boundary conditions


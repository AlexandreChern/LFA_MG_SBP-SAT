## On the $\omega$ values for the SBP-SAT methods


The test has shown that omega values need to be larger than 1 for the SOR method to get accelerated convergence. Ideal $omega$ values should be around 1.9. The residual plots show unsmoothed error distribution.


Notes from April 6
Using more iterations on the for smoothers would accelerate the MG performance significantly. Once steps of smoothers are enough to smooth results on different levels. The performance of the SBP-SAT MG matches the performance of the central finite difference with inject with the same MG configurations.


The bad performance of MG for the SBP-SAT method is very likely coming from the smoothers. The Jacobi / Gauss-Seidel method is not a good smoother on the coarsest grid. While they perform well with the central finite difference with injected boundary conditions



Notes from April 7

Arpack package provides eigvals solver for sparse format eigs


Notes from April 15

We need to use Galerkin condition to form the coarse grid operators in order to have MG converges. However, this will make the matrix-free code more challenging to write.
Alex


Notes from April 20

1. Test difference between using Galerkin conditions and not using Galerkin conditions
2. Test Operator-dependent interpolations to see if we can get improvements in convergence rate for GMG


Notes from April 22

1. Using operator-dependnt interpolations do help with the convergence for 1 MG iteration
2. The performace is not ideal after more iterations. Need to form prolongation operators better


Notes from April 24

1. A_mg formed with Galerkin conditions look like centered finite difference operators

2. Try to form this directly without using Galerkin conditions


Notes from April 25

1. Find the issue with Geometric Multigrid
2. The H_tilde matrix is different on different grids. To solve the problem directly, it's okay. But for multigrid, it causes different coefficients on the LHS and RHS of the equation. The right-hand side is interpolated, so the H_tilde is associated with the finiest grid. The left-hand-side is formed directly, so the H_tilde will get larger and larger on the left hand side.
3. This is why Algebraic multigrid was working. Because the LHS operator is obtained through interpolation, so the H_tilde does not grow with coarse grid refinements


Notes from April 16

1. Adjusting a constant coefficients to H_tilde does help with residual, but not helping with the error.

2. Removing H_tilde entirely is ideal on smaller problems but not on large problems with more iteration. Probably okay as a preconditioner.
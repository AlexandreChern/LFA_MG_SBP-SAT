using LinearAlgebra
using SparseArrays

function red_black_sor(A, b, x0, w, tol, maxiter)
    n = length(b)
    x = copy(x0)
    err = Inf
    iter = 0
    omega = w / 2.0
    
    while err > tol && iter < maxiter
        err = 0.0
        for color in (1,2)
            for i in color:n
                s = dot(A[i,:], x) - A[i,i] * x[i]
                x[i] += omega * (b[i] - s) / A[i,i]
            end
        end
        for color in (2,1)
            for i in color:n
                s = dot(A[i,:], x) - A[i,i] * x[i]
                x[i] += omega * (b[i] - s) / A[i,i]
            end
        end
        err = norm(b - A*x)
        iter += 1
    end
    return x, iter
end

# Example system to solve: Ax = b
A = [4.0 -1.0 0.0 -1.0 0.0 0.0;
    -1.0 4.0 -1.0 0.0 -1.0 0.0;
     0.0 -1.0 4.0 0.0 0.0 -1.0;
    -1.0 0.0 0.0 4.0 -1.0 0.0;
     0.0 -1.0 0.0 -1.0 4.0 -1.0;
     0.0 0.0 -1.0 0.0 -1.0 4.0]
b = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Initial guess
x0 = zeros(length(b))

# Parameters
w = 0.5
tol = 1e-6
maxiter = 1000

# Solve using red-black SOR algorithm
x, iter = red_black_sor(A, b, x0, w, tol, maxiter)

# Print solution and number of iterations
println("Solution: ", x)
println("Number of iterations: ", iter)
include("diagonal_sbp.jl")


# Solving Poisson Equation
# - Δu(x,y) = f(x,y)
# Manufactured Solution: u(x,y) = (x^2 -1) * (y^2 - 1) on a unit square
# source term f(x,y) = -2(2 - x^2 - y^2)

# using CuArrays, CUDAnative
using LinearAlgebra
using SparseArrays
using Plots
using IterativeSolvers
using BenchmarkTools
using MAT

function e(i,n)
    # A = Matrix{Float64}(I,n,n)
    # return A[:,i]
    out = spzeros(n)
    out[i] = 1.0
    return out 
end

function eyes(n)
    # return Matrix{Float64}(I,n,n)
    out = spzeros(n,n)
    for i in 1:n
        out[i,i] = 1.0
    end
    return out
end

function u(x,y)
    # return sin.(π*x .+ π*y)
    return (x.^2 .- 1) .* (y.^2 .- 1)
end

function f(x,y)
    return 2 .* (2 .- x.^2 .- y.^2)
end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end

function Operators_2d(i, j, p=2, h_list_x = ([1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]),
			 h_list_y = ([1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13])
			 )
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    # Matrix Size
    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,N_x,xc=(0,1));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,N_x,xc=(0,1));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,N_y,xc=(0,1));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,N_y,xc=(0,1));

    # BSx = sparse(SNx - S0x);
    # BSy = sparse(SNy - S0y);
    BSx = SNx - S0x
    BSy = SNy - S0y

    # Forming 2d Operators
    # e_1x = sparse(e(1,N_x+1));
    # e_Nx = sparse(e(N_x+1,N_x+1));
    # e_1y = sparse(e(1,N_y+1));
    # e_Ny = sparse(e(N_y+1,N_y+1));
    e_1x = e(1,N_x+1);
    e_Nx = e(N_x+1,N_x+1);
    e_1y = e(1,N_x+1);
    e_Ny = e(N_y+1,N_y+1);

    # I_Nx = sparse(eyes(N_x+1));
    # I_Ny = sparse(eyes(N_y+1));
    I_Nx = eyes(N_x+1);
    I_Ny = eyes(N_y+1);


    e_E = kron(e_Nx,I_Ny);
    e_W = kron(e_1x,I_Ny);
    e_S = kron(I_Nx,e_1y);
    e_N = kron(I_Nx,e_Ny);

    E_E = kron(sparse(Diag(e_Nx)),I_Ny);   # E_E = e_E * e_E'
    E_W = kron(sparse(Diag(e_1x)),I_Ny);
    E_S = kron(I_Nx,sparse(Diag(e_1y)));
    E_N = sparse(kron(I_Nx,sparse(Diag(e_Ny))));


    D1_x = kron(D1x,I_Ny);
    D1_y = kron(I_Nx,D1y);


    D2_x = kron(D2x,I_Ny);
    D2_y = kron(I_Nx,D2y);
    D2 = D2_x + D2_y


    HI_x = kron(HIx,I_Ny);
    HI_y = kron(I_Nx,HIy);

    H_x = kron(H1x,I_Ny);
    H_y = kron(I_Nx,H1y);

    BS_x = kron(BSx,I_Ny);
    BS_y = kron(I_Nx,BSy);


    HI_tilde = kron(HIx,HIx);
    H_tilde = kron(H1x,H1y);

    return (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N)
end

h_list_x = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]
h_list_y = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]

rel_errs = []
iter_errs = []
# for k in 1:length(h_list_x)
println("################### BEGIN TEST #########################")
for k in 6:7
    
    println()
    i = j  = k
    println("##########   Starting Test for k = ", k, "   ######################")
   
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    # Matrix Size
    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    Nx = N_x + 1;
    Ny = N_y + 1;

    println("Nx = $Nx, Ny=$Ny")

    # 2D operators
    (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j);


     # Analytical Solutions
    analy_sol = u(x,y');

   

    # Forming SAT terms (D,D,D,D)

    # Penalty Parameters
    tau_E = 13/hx;
    tau_W = 13/hx;
    tau_N = 13/hy;
    tau_S = 13/hy;
 
    beta = 1;

    ## Formulation 1
    SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W;
    SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E;
    SAT_S = tau_S*HI_y*E_S + beta*HI_y*BS_y'*E_S;
    SAT_N = tau_N*HI_y*E_N + beta*HI_y*BS_y'*E_N;

    SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W;
    SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E;
    SAT_S_r = tau_S*HI_y*E_S*e_S + beta*HI_y*BS_y'*E_S*e_S;
    SAT_N_r = tau_N*HI_y*E_N*e_N + beta*HI_y*BS_y'*E_N*e_N;


    (alpha1,alpha2,alpha3,alpha4,beta) = (tau_N,tau_S,tau_W,tau_E,beta);


    g_W = -1 * (y.^2 .- 1);
    g_E = (0) * (y.^2 .- 1) ;
    g_S = (x.^2 .- 1) * (-1);
    g_N = (x.^2 .- 1) * (0);

    # Solving with CPU
    A = -D2 + SAT_W + SAT_E + SAT_S + SAT_N;

    b = f(x,y')[:] + SAT_W_r*g_W + SAT_E_r*g_E + SAT_S_r*g_S + SAT_N_r*g_N;

    A_DDDD = H_tilde*A;
    b_DDDD = H_tilde*b;

    direct_sol_DDDD = A_DDDD\b_DDDD
    direct_sol_matrix_DDDD = reshape(direct_sol,Nx,Ny)

    surface(x,y,direct_sol_matrix_DDDD)

    ## Formulation 2 (D,D,N,N)

    tau_E = 13/hx;
    tau_W = 13/hx;
    tau_N = 1;
    tau_S = 1;

    beta = 1;

    SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W;
    SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E;
    
    # SAT_S = tau_S*HI_y*E_S*D1_y
    # SAT_N = tau_N*HI_y*E_N*D1_y

    SAT_S = tau_S*HI_y*E_S*BS_y;
    SAT_N = tau_N*HI_y*E_N*BS_y;

    SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W;
    SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E;
    SAT_S_r = tau_S*HI_y*E_S*e_S;
    SAT_N_r = tau_N*HI_y*E_N*e_N;


    (alpha1,alpha2,alpha3,alpha4,beta) = (tau_N,tau_S,tau_W,tau_E,beta);


    g_W = -1 * (y.^2 .- 1);
    g_E = (0) * (y.^2 .- 1) ;
    g_S = (x.^2 .- 1) .* (-2 .* 0);
    # g_N = -π*cos.(π*x)
    g_N = (x.^2 .- 1) .* (2 .* 1);

    # Solving with CPU
    A = -D2 + SAT_W + SAT_E + SAT_S + SAT_N;

    b = f(x,y')[:] + SAT_W_r*g_W + SAT_E_r*g_E + SAT_S_r*g_S + SAT_N_r*g_N;

    A_DDNN = - H_tilde*A;
    b_DDNN = - H_tilde*b;

    direct_sol_DDNN = A_DDNN\b_DDNN
    direct_sol_matrix_DDNN = reshape(direct_sol,Nx,Ny)

    surface(x,y,direct_sol_matrix_DDNN)


    u_exact = u(x,y')
    surface(x,y,u_exact)

    @show nnz(A) * sizeof(Float64)

    @show Base.summarysize(A)
    @show Base.summarysize(b)
    @show Nx * Ny * sizeof(Float64)

    @show Base.summarysize(A) / Base.summarysize(b)
    
    println()


    ## SpMV test
   
    println("TESTING SpMV vs Matrix_FREE")
    iter_times = Nx + Ny



    ## End Compare Efficiency
    println("##########   Ending Test for k = ", k, "   ######################\n")

   
end

function check_richardson(A,ω)
    return opnorm(Matrix(I,size(A)) - ω*A)
end


function modified_richardson(A,b;ω=0.15)
    x = zeros(length(b))
    # ω = -0.15
    count = 0
    norms = []
    for _ in 1:100
        iter_norm = norm(A*x-b)
        append!(norms,iter_norm)
        if iter_norm >= 1e-6 * norms[1]
            count += 1
            x = x + ω*(b- A*x)
        end
    end
    return (x,count,norms)
end


function jacobi_diy(A,b)
    x = zeros(length(b))
    D = Diagonal(A)
    LU = A - D
    # ω = -0.15
    count = 0
    norms = []
    for _ in 1:100
        iter_norm = norm(A*x-b)
        append!(norms,iter_norm)
        if iter_norm >= 1e-6 * norms[1]
            count += 1
            x = D\(b - LU*x)
        end
    end
    return (x,count,norms)
end

function test(A,b;ω=1/13)
    (x_j,count_j,norms_j) = jacobi_diy(A,b)
    (x_r,count_r,norms_r) = modified_richardson(A,b,ω=ω)
    plot(norms_j)
    plot!(norms_r)
end
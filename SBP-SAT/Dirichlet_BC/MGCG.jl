include("mg_matrix_N.jl")


mutable struct MG
    A_mg
    L_mg
    U_mg
    f_mg
    u_mg
    r_mg
    rest_mg
    prol_mg
    lnx_mg
    lny_mg
end

mg_struct = MG([],[],[],[],[],[],[],[],[],[])

function clear_mg_struct(mg_struct)
    mg_struct.A_mg = []
    mg_struct.L_mg = []
    mg_struct.U_mg = []
    mg_struct.f_mg = []
    mg_struct.u_mg = []
    mg_struct.r_mg = []
    mg_struct.rest_mg = []
    mg_struct.prol_mg = []
    mg_struct.lnx_mg = []
    mg_struct.lny_mg = []
end

function initialize_mg_struct(mg_struct,nx,ny,n_level;use_galerkin=false,use_sbp=true)
    println("clearing matrices")
    clear_mg_struct(mg_struct)
    println("Starting assembling matrices")
    A_mg = mg_struct.A_mg
    L_mg = mg_struct.L_mg
    U_mg = mg_struct.U_mg
    f_mg = mg_struct.f_mg
    u_mg = mg_struct.u_mg
    r_mg = mg_struct.r_mg
    rest_mg = mg_struct.rest_mg
    prol_mg = mg_struct.prol_mg
    lnx_mg = mg_struct.lnx_mg
    lny_mg = mg_struct.lny_mg
    if isempty(A_mg) 
        # Assembling matrices
        for k in 1:n_level
            nx,ny = nx,ny
            hx,hy = 1/nx, 1/ny
            if k == 1
                A_DDDD, b_DDDD = poisson_sbp_sat_matrix(nx,ny,1/nx,1/ny)
                push!(A_mg,A_DDDD)
                push!(f_mg,reshape(b_DDDD,nx+1,ny+1))
            else
                if use_galerkin
                    A_DDDD = rest_mg[k-1] * A_mg[k-1] * prol_mg[k-1]
                else
                    A_DDDD, b_DDDD = poisson_sbp_sat_matrix(nx,ny,1/nx,1/ny)
                end
                push!(A_mg,A_DDDD)
                push!(f_mg, spzeros(nx+1,ny+1))
            end
            # push!(L_mg, LowerTriangular(A_mg[k]))
            push!(L_mg, tril(A_mg[k],0))
            push!(U_mg, triu(A_mg[k],1))

            push!(u_mg, spzeros(nx+1,ny+1))
            push!(r_mg, spzeros(nx+1,ny+1))
            if use_sbp
                push!(rest_mg, restriction_matrix_v2(nx,ny,div(nx,2),div(ny,2)))
                # push!(prol_mg, prolongation_matrix_v2(nx,ny,div(nx,2),div(ny,2)))
                push!(prol_mg, 4*restriction_matrix_v2(nx,ny,div(nx,2),div(ny,2))')
            else
                push!(rest_mg, restriction_matrix_v1(nx,ny,div(nx,2),div(ny,2)))
                push!(prol_mg, prolongation_matrix_v1(nx,ny,div(nx,2),div(ny,2)))
            end
            push!(lnx_mg,nx)
            push!(lny_mg,ny)
            nx,ny = div(nx,2), div(ny,2)
            hx,hy = 2*hx, 2*hy
        end
    end
    println("Ending resembling matrices")
end

function mg_solver(mg_struct, f_in ;nx=64,ny=64,n_level=3,v1=2,v2=2,v3=2,tolerance=1e-10,iter_algo_num=1,interp="normal",ω=1,maximum_iterations=10,use_galerkin=false,use_sbp=true)
    initialize_mg_struct(mg_struct,nx,ny,n_level,use_galerkin=use_galerkin,use_sbp=use_sbp)
    # mg_struct.u_mg[1][:] .= u_in
    mg_struct.f_mg[1][:] .= copy(f_in)[:]
    mg_struct.u_mg[1][:] .= spzeros(nx+1,ny+1)[:]
    # ω = 1 # damping coefficient for SOR
    iter_algos = ["gauss_seidel","SOR","jacobi","chebyshev","richardson"]
    iter_algo = iter_algos[iter_algo_num]
    # maximum_iterations = 120 #nx*ny # set maximum_iterations
    # compute the initial residual
    mg_struct.r_mg[1][:] = mg_struct.f_mg[1][:] - mg_struct.A_mg[1] * mg_struct.u_mg[1][:]
    dx = 1.0 ./nx
    dy = 1.0 ./ny
    xs = 0:dx:1
    ys = 0:dy:1
    u_exact = (xs.^2 .- 1) .* (ys'.^2 .- 1)

    # compute initial L-2 norm
    rms = compute_l2norm(nx,ny,mg_struct.r_mg[1])
    init_rms = rms
    mg_iter_count = 0
    println("0", " ", rms, " ", rms/init_rms)
    if nx < (2^n_level)
        println("Number of levels exceeds the possible number.")
    end

    # Allocate matrix for storage at fine level
    # residual at fine level is already defined at global level
    prol_fine = zeros(Float64, mg_struct.lnx_mg[1]+1, mg_struct.lny_mg[1]+1)

    # temporary residual which is restricted to coarse mesh error
    # the size keeps on changing

    temp_residual = zeros(Float64, mg_struct.lnx_mg[1]+1, mg_struct.lny_mg[1]+1)

    # u_n .= reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
    # reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

    println("Starting Multigrid Iterations")
    for iteration_count = 1:maximum_iterations
        mg_iter_count += 1
        # @show mg_iter_count

        # starting pre-smoothing on the finest grid
        for i in 1:v1
            if iter_algo == "gauss_seidel"
                # u_mg[1] .= reshape(L_mg[1]\(f_mg[1][:] .- U_mg[1]*u_mg[1][:]), nx+1, ny+1)
                mg_struct.u_mg[1][:] .= mg_struct.L_mg[1] \ (mg_struct.f_mg[1][:] .- mg_struct.U_mg[1]*mg_struct.u_mg[1][:])
            elseif iter_algo == "SOR"
                # u_mg[1][:] .= (1-ω) * u_mg[1][:] .+ ω * L_mg[1]\(f_mg[1][:] .- U_mg[1]*u_mg[1][:])
                mg_struct.u_mg[1][:] .= sor!(mg_struct.u_mg[1][:],mg_struct.A_mg[1],mg_struct.f_mg[1][:],ω;maxiter=1)
            elseif iter_algo == "jacobi"
                mg_struct.u_mg[1][:] .= ω * jacobi!(mg_struct.u_mg[1][:],mg_struct.A_mg[1],mg_struct.f_mg[1][:],maxiter=1) .+ (1-ω) *  mg_struct.u_mg[1][:]
            elseif iter_algo == "chebyshev"
                mg_struct.u_mg[1][:] .= chebyshev!(mg_struct.u_mg[1][:],mg_struct.A_mg[1],mg_struct.f_mg[1][:],0,40,maxiter=1)
            elseif iter_algo == "richardson"
                mg_struct.u_mg[1][:] .= mg_struct.u_mg[1][:] .+ ω * (mg_struct.f_mg[1][:] .- mg_struct.A_mg[1]*mg_struct.u_mg[1][:])
            end
        end

        # decending from the second level to the coarsest level
        for k = 2:n_level
            if k == 2
                # for the second level temporary residual is take from the finest mesh
                mg_struct.r_mg[k-1] = mg_struct.r_mg[1]
            else
                mg_struct.r_mg[k-1][:] .= mg_struct.f_mg[k-1][:] - mg_struct.A_mg[k-1] * mg_struct.u_mg[k-1][:]
            end
            mg_struct.f_mg[k] = mg_struct.rest_mg[k-1] * mg_struct.r_mg[k-1][:]

            # smoothing on k level when k < n_level
            if k < n_level
                for i in 1:v1
                    if iter_algo == "gauss_seidel"
                        # u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] .- U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1) # gauss seidel
                        mg_struct.u_mg[k][:] .= mg_struct.L_mg[k] \ (mg_struct.f_mg[k][:] .- mg_struct.U_mg[k] * mg_struct.u_mg[k][:])
                    elseif iter_algo == "SOR"
                        # u_mg[k][:] = (1-ω) * u_mg[k][:] .+ ω * L_mg[k] \ (f_mg[k][:] .- U_mg[k]*u_mg[k][:]) # SOR
                        mg_struct.u_mg[k][:] .= sor!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],ω;maxiter=1)
                    elseif iter_algo == "jacobi"
                        mg_struct.u_mg[k][:] .= ω * jacobi!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],maxiter=1) .+ (1-ω) * mg_struct.u_mg[k][:]
                    elseif iter_algo == "chebyshev"
                        mg_struct.u_mg[k][:] .= chebyshev!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],0,40,maxiter=1)
                    elseif iter_algo == "richardson"
                        mg_struct.u_mg[k][:] .= mg_struct.u_mg[k][:] .+ ω * (mg_struct.f_mg[k][:] .- mg_struct.A_mg[k]*mg_struct.u_mg[k][:])
                    end
                end
            # reaching the coarsest grid
            # we can decide to do smoothing here as well
            # or use solvers such as CG
            elseif k == n_level
                for i in 1:v2
                    if iter_algo == "gauss_seidel"
                        # u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] .- U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1) # gauss seidel
                        mg_struct.u_mg[k][:] .= mg_struct.L_mg[k] \ (mg_struct.f_mg[k][:] .- mg_struct.U_mg[k] * mg_struct.u_mg[k][:])
                    elseif iter_algo == "SOR"
                        # u_mg[k][:] = (1-ω) * u_mg[k][:] .+ ω * L_mg[k] \ (f_mg[k][:] .- U_mg[k]*u_mg[k][:]) # SOR
                        mg_struct.u_mg[k][:] .= sor!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],ω;maxiter=1)
                        # mg_struct.u_mg[k][:] .= cg!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:]) # solve using CG
                    elseif iter_algo == "jacobi"
                        mg_struct.u_mg[k][:] .= ω * jacobi!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],maxiter=1) .+ (1-ω) * mg_struct.u_mg[k][:]
                    elseif iter_algo == "chebyshev"
                        mg_struct.u_mg[k][:] .= chebyshev!(mg_struct.u_mg[k][:],mg_struct.A_mg[k],mg_struct.f_mg[k][:],0,40,maxiter=1)
                    elseif iter_algo == "richardson"
                        mg_struct.u_mg[k][:] .= mg_struct.u_mg[k][:] .+ ω * (mg_struct.f_mg[k][:] .- mg_struct.A_mg[k]*mg_struct.u_mg[k][:])
                    end
                end
            end
        end

        # ascending from the coarsest grid to the finiest grid
        for k = n_level-1:-1:2
            # temporary matrix for correction storage at the (k-1)th level
            # solution prolongated from the kth level to the (k-1)th level
            prol_fine = zeros(Float64, mg_struct.lnx_mg[k-1]+1, mg_struct.lny_mg[k-1]+1)

            # prolongate solution from (k)th level to (k-1)th level
            # prol_fine[:] = prolongation_matrix(lnx[k-1],lny[k-1],lnx[k],lny[k]) * u_mg[k][:]
            prol_fine[:] = mg_struct.prol_mg[k-1] * mg_struct.u_mg[k][:]

            # update u_mg
            for j = 1:mg_struct.lnx_mg[k-1]+1 for i in 1:mg_struct.lnx_mg[k-1]+1
                mg_struct.u_mg[k-1][i,j] = mg_struct.u_mg[k-1][i,j] + prol_fine[i,j]
            end end

            # post smoothing
            for i in 1:v3
                if iter_algo == "gauss_seidel"
                    # u_mg[k-1] .= reshape(L_mg[k-1]\(f_mg[k-1][:] .- U_mg[k-1]*u_mg[k-1][:]),lnx[k-1]+1,lny[k-1]+1)
                    mg_struct.u_mg[k-1][:] .= mg_struct.L_mg[k-1] \ (mg_struct.f_mg[k-1][:] .- mg_struct.U_mg[k-1] * mg_struct.u_mg[k-1][:])
                elseif iter_algo == "SOR"
                    # u_mg[k-1][:] = (1-ω) * u_mg[k-1][:] .+ ω * L_mg[k-1]\(f_mg[k-1][:] .- U_mg[k-1]*u_mg[k-1][:]) 
                    mg_struct.u_mg[k-1][:] .= sor!(mg_struct.u_mg[k-1][:],mg_struct.A_mg[k-1],mg_struct.f_mg[k-1][:],ω,maxiter=1)
                elseif iter_algo == "jacobi"
                    mg_struct.u_mg[k-1][:] .= ω * jacobi!(mg_struct.u_mg[k-1][:],mg_struct.A_mg[k-1],mg_struct.f_mg[k-1][:],maxiter=1) .+ (1-ω) * mg_struct.u_mg[k-1][:]
                elseif iter_algo == "chebyshev"
                    mg_struct.u_mg[k-1][:] .= chebyshev!(mg_struct.u_mg[k-1][:],mg_struct.A_mg[k-1],mg_struct.f_mg[k-1][:],0,40,maxiter=1)
                elseif iter_algo == "richardson"
                    mg_struct.u_mg[k-1][:] .= mg_struct.u_mg[k-1][:] .+ ω * (mg_struct.f_mg[k-1][:] .- mg_struct.A_mg[k-1] * mg_struct.u_mg[k-1][:])
                end
            end
        end
        mg_struct.r_mg[1][:] .= mg_struct.f_mg[1][:] .- mg_struct.A_mg[1] * mg_struct.u_mg[1][:]
        # mg_struct.f_mg[1][:] .= copy(f_in)[:]
        # if iteration_count % 5 == 0
        #     contourf(xs,ys,r,levels=20,color=:turbo)
        #     savefig("figures/$(iteration_count)_res.png")
        #     contourf(xs,ys,u_mg[1]-u_exact,levels=20,color=:turbo)
        #     savefig("figures/$(iteration_count)_error.png")
        # end

        rms = compute_l2norm(mg_struct.lnx_mg[1],mg_struct.lny_mg[1],mg_struct.r_mg[1])
        error = norm(mg_struct.u_mg[1] - u_exact)
        println("$(iteration_count)", " rms ", rms, " rms/init_rms ", rms/init_rms, " log(rms) ", log(rms))
        println("$(iteration_count)", " error ", error, " log(error) ", log(error))

    end
    return mg_struct.u_mg[1]
end



function mgcg(mg_struct;nx=64,ny=64,n_level=3,v1=2,v2=2,v3=2,ω=1.8, maxiter=100,iter_algo_num=2,maximum_iterations=2,precond=true)
    x = spzeros(nx+1,ny+1)
    r = spzeros(size(x))
    A,b = poisson_sbp_sat_matrix(nx,ny,1/nx,1/ny)
    r[:] .= b[:] .- A * x[:]
    init_rms = norm(r)
    @show init_rms
    z = spzeros(size(r));
    if precond == true
        z .= mg_solver(mg_struct, r, n_level=n_level, v1=v1,v2=v2,v3=v3, maximum_iterations=maximum_iterations, nx=nx, ny=ny, iter_algo_num=iter_algo_num);
    else
        z[:] .= r[:]
    end
    p = spzeros(size(r));
    p .= z ;
    counter = 0
    for k in 1:maxiter
        α = dot(r[:],z[:]) / (dot(p[:],A*p[:]))
        x .= x .+ α * p
        r_new = r[:] .- α * A * p[:]
        @show norm(r_new) norm(r_new) / init_rms
        if norm(r_new) < 1e-8 * init_rms
            break
        end
        if precond == true
            z_new = mg_solver(mg_struct, r_new, n_level=n_level, v1=v1,v2=v2,v3=v3, maximum_iterations=maximum_iterations, nx=nx, ny=ny, iter_algo_num=iter_algo_num)
        else
            z_new = copy(r_new)
        end
        β = dot(r_new[:],z_new[:]) / (dot(r[:],z[:]))
        p[:] .= z_new[:] .+ β * p[:]
        z[:] .= z_new[:]
        r[:] .= r_new[:]
        counter += 1
    end
    return x, counter
end

function test_mgcg()
    # u0 = randn(nx+1,ny+1)
    mgcg(mg_struct,nx=128,ny=128,maxiter=100,iter_algo_num=2)
    mgcg(mg_struct,nx=512,ny=512,n_level=8,iter_algo_num=1,maxiter=1000,precond=false)
    mgcg(mg_struct,nx=512,ny=512,n_level=8,v1=4,v2=4,v3=10,iter_algo_num=1,maxiter=1000,precond=true)
    mgcg(mg_struct,nx=512,ny=512,n_level=8,v1=4,v2=4,v3=10,ω=1.6,iter_algo_num=2,maxiter=1000,precond=true)
    mgcg(mg_struct,nx=512,ny=512,n_level=8,v1=4,v2=4,v3=10,ω=1.6,iter_algo_num=2,maxiter=1000,precond=true)
    mg_solver(mg_struct, b_64, nx=64,ny=64,n_level=6,v1=10,v3=10,v2=10,iter_algo_num=1,use_galerkin=true,maximum_iterations=40,use_sbp=true) # this works well
end

function initial_global_params()
    nx=64
    ny=64
    n_level=3
    v1=2
    v2=2
    v3=2
    tolerance=1e-10
    iter_algo_num=2
    interp="sbp"
    ω=1
    maximum_iterations=120
end


let
    _, b_8 = poisson_sbp_sat_matrix(8,8,1/8,1/8) 
    _, b_16 = poisson_sbp_sat_matrix(16,16,1/16,1/16) 
    _, b_32 = poisson_sbp_sat_matrix(32,32,1/32,1/32) 
    _, b_64 = poisson_sbp_sat_matrix(64,64,1/64,1/64) 
    _, b_128 = poisson_sbp_sat_matrix(128,128,1/128,1/128) 
    _, b_256 = poisson_sbp_sat_matrix(256,256,1/256,1/256) 
    _, b_512 = poisson_sbp_sat_matrix(512,512,1/512,1/512)
    _, b_1024 = poisson_sbp_sat_matrix(1024,1024,1/1024,1/1024)  
end
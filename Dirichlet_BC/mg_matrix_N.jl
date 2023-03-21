using CPUTime
using Printf
using Plots
using SparseArrays
using LinearAlgebra

function compute_residual(nx,ny,dx,dy,f,u_n,r)
    for j in 2:ny for i = 2:nx
        d2udx2 = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx^2)
        d2udy2 = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy^2)
        r[i,j] = f[i,j]  - d2udx2 - d2udy2
        # r[i,j] = d2udx2 + d2udy2
    end end
end


function poisson_matrix_old(nx,ny,dx,dy)
    poisson_matrix_ = spzeros((nx+1)*(ny+1),(nx+1)*(ny+1))
    Dxy = -2 * (1/dx^2 + 1/dy^2)
    Dx = 1 / dx^2
    Dy = 1 / dy^2
    for j in 1:(nx+1)*(ny+1) for i in 1:(nx+1)*(ny+1)
        if i == j
            poisson_matrix_[i,j] = (Dxy)
        end
        if (j == i + 1) || (j == i - 1)
            poisson_matrix_[i,j] = Dy
        end
        if (j == i + ny + 1 || j == i - ny - 1)
            poisson_matrix_[i,j] = Dx
        end
    end end

    index_count = 0
    for j in 1:ny+ 1 for i in 1:nx + 1
        index = (j - 1) * (nx+1) + i
        if (i == 1 || i == nx + 1 || j == 1 || j == ny + 1)
            poisson_matrix_[index,:] .= 0
            poisson_matrix_[index,index] = 1
            index_count += 1
        end
    end end
    # @show index_count
    return poisson_matrix_
end


function poisson_matrix(nx,ny,dx,dy)
    Dx = 1 / dx^2
    Dy = 1 / dy^2

    Dxx_matrix_ = spzeros(nx+1,nx+1)
    # Dxx_matrix_[1,1] = -2
    Dxx_matrix_[end,end] = -2 * Dx
    for i in 1:nx
        Dxx_matrix_[i,i] = -2 * Dx
        Dxx_matrix_[i,i+1] = 1 * Dx
        Dxx_matrix_[i+1,i] = 1 * Dx
    end

    Dyy_matrix_ = spzeros(ny+1,ny+1)
    # Dxx_matrix_[1,1] = -2
    Dyy_matrix_[end,end] = -2 * Dy
    for i in 1:nx
        Dyy_matrix_[i,i] = -2 * Dy
        Dyy_matrix_[i,i+1] = 1 * Dy
        Dyy_matrix_[i+1,i] = 1 * Dy
    end
    I_Ny = sparse(I,ny+1,ny+1)
    I_Nx = sparse(I,nx+1,nx+1)
    poisson_matrix_ = kron(I_Nx,Dyy_matrix_) + kron(Dxx_matrix_,I_Ny)
    index_count = 0
    for j in 1:ny+ 1 for i in 1:nx + 1
        index = (j - 1) * (nx+1) + i
        if (i == 1 || i == nx + 1 || j == 1 || j == ny + 1)
            poisson_matrix_[index,:] .= 0
            poisson_matrix_[index,index] = 1
            index_count += 1
        end
    end end
    # @show index_count
    return poisson_matrix_
end

function compute_l2norm(nx, ny, r)
    rms = 0.0
    # println(residual)
    for j = 2:ny for i = 2:nx
        rms = rms + r[i,j]^2
    end end
    # println(rms)
    rms = sqrt(rms/((nx-1)*(ny-1)))
    return rms
end




"""
    restriction_matrix(nxf,nyf,nxc,nyc)
    Generating restriction matrix from fine grid (nxf,nyf) to coarse grid (nxc, nyc)
    The matrix size generated is ((nxc+1) * (nyc+1) by (nxf+1) * (nyf+1))
    
    # Examples
    ```julia
    julia> restriction_matrix(4,4,2,2)
    9×25 SparseMatrixCSC{Float64, Int64} with 25 stored entries:
    ⠑⠒⢄⠀⠀⡀⠀⢀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠉⠑⠈⠉⠂⠉⠑⢄⣀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁
    ```
"""
function restriction_matrix(nxf,nyf,nxc,nyc)
    restriction_matrix_ = spzeros((nxc+1)*(nyc+1),(nxf+1)*(nyf+1))
    for j in 1:nyc+1
        for i in 1:nxc+1
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            # @show (indexc, indexf)
            restriction_matrix_[indexc,indexf] = (4)/16.0
            if 2 <= i <= nxc
                restriction_matrix_[indexc,indexf+1] = (2)/16.0
                restriction_matrix_[indexc,indexf-1] = (2)/16.0
            end
            if 2 <= j <= nxc
                restriction_matrix_[indexc,indexf+nyf+1] = (2)/16.0
                restriction_matrix_[indexc,indexf-nyf-1] = (2)/16.0
            end
            if (2 <= i <= nxc) && (2 <= j <= nyc)
                restriction_matrix_[indexc,indexf-nyf-1-1] = (1)/16.0
                restriction_matrix_[indexc,indexf-nyf-1+1] = (1)/16.0
                restriction_matrix_[indexc,indexf+nyf+1+1] = (1)/16.0
                restriction_matrix_[indexc,indexf+nyf+1-1] = (1)/16.0
            end
        end
    end
    return restriction_matrix_
end


"""
    prolongation_matrix(nxf,nyf,nxc,nyc)
    Generating restriction matrix from coarse grid (nxc, nyc) to fine grid (nxf,nyf)
    The matrix size generated is (nxf+1) * (nyf+1)) by ((nxc+1) * (nyc+1) 

    # Examples
    julia> prolongation_matrix(4,4,2,2)
    25×9 SparseMatrixCSC{Float64, Int64} with 25 stored entries:
    ⢱⠀⠀⠀⠀
    ⠀⠑⡄⠀⠀
    ⠀⠠⡑⠀⠀
    ⠀⢀⠣⠀⠀
    ⠀⠀⢇⠀⠀
    ⠀⠀⠀⢱⠀
    ⠀⠀⠀⠀⠁
"""
function prolongation_matrix(nxf,nyf,nxc,nyc)
    prolongation_matrix_ = spzeros((nxf+1)*(nyf+1),(nxc+1)*(nyc+1))
    for j in 1:nyc+1
        for i in 1:nxc+1
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            # @show (indexc, indexf)
            prolongation_matrix_[indexf,indexc] = 1.0  # direct injection instead of(4)/16.0
            if 2 <= i <= nxc
                prolongation_matrix_[indexf+1,indexc] = 0.5 #(2)/16.0
                prolongation_matrix_[indexf-1,indexc] = 0.5 #(2)/16.0
            end
            if 2 <= j <= nxc
                prolongation_matrix_[indexf+nyf+1,indexc] = 0.5 #(2)/16.0
                prolongation_matrix_[indexf-nyf-1,indexc] = 0.5 #(2)/16.0
            end
            if (2 <= i <= nxc) && (2 <= j <= nyc)
                prolongation_matrix_[indexf-nyf-1-1,indexc] = 0.25 #(1)/16.0
                prolongation_matrix_[indexf-nyf-1+1,indexc] = 0.25 #(1)/16.0
                prolongation_matrix_[indexf+nyf+1+1,indexc] = 0.25 #(1)/16.0
                prolongation_matrix_[indexf+nyf+1-1,indexc] = 0.25 #(1)/16.0
            end
        end
    end
    return prolongation_matrix_
end




# function initialize_uf(u_n, f_array)
function initialize_uf(nx,ny;ipr=1)
    # u_n = Array{Float64}(undef, nx+1, ny+1)
    # nx, ny = size(u_n)[1]-1, size(u_n)[2]-1
    u_n = Array{Float64}(undef,nx+1,ny+1)
    f_array = Array{Float64}(undef,nx+1,ny+1)
    u_e = Array{Float64}(undef, nx+1, ny+1)

    x_l = 0.0
    x_r = 1.0
    y_b = 0.0
    y_t = 1.0

    dx = (x_r - x_l) / nx
    dy = (y_t - y_b) / ny

    x = Array{Float64}(undef, nx+1)
    y = Array{Float64}(undef, ny+1)
    u_e = Array{Float64}(undef, nx+1, ny+1)
    f = Array{Float64}(undef, nx+1, ny+1)
    u_n = Array{Float64}(undef, nx+1, ny+1)

    for i = 1:nx+1
        x[i] = x_l + dx*(i-1)
    end
    for i = 1:ny+1
        y[i] = y_b + dy*(i-1)
    end

    for i = 1:nx+1 for j = 1:ny+1
        if ipr == 1
            u_e[i,j] = (x[i]^2 - 1.0)*(y[j]^2 - 1.0)
    
            f_array[i,j]  = -2.0*(2.0 - x[i]^2 - y[j]^2)
    
            u_n[i,j] = 0.0
        end
    
        if ipr == 2
            u_e[i,j] = sin(2.0*pi*x[i]) * sin(2.0*pi*y[j]) +
                       c1*sin(16.0*pi*x[i]) * sin(16.0*pi*y[j])
    
            f_array[i,j] = 4.0*c2*sin(2.0*pi*x[i]) * sin(2.0*pi*y[j]) +
                     c2*sin(16.0*pi*x[i]) * sin(16.0*pi*y[j])
    
            u_n[i,j] = 0.0
        end
    end end

    u_n[:,1] = u_e[:,1]
    u_n[:, ny+1] = u_e[:, ny+1]

    u_n[1,:] = u_e[1,:]
    u_n[nx+1,:] = u_e[nx+1,:]

    for i = 1:nx+1 for j = 1:ny+1
        if ((i == 1) || (i == nx+1) || (j == 1) || (j == ny+1))
            f_array[i,j] = u_n[i,j]
        end
    end end

    return u_n, f_array
end



#######################################################################
## Starting multigrid

function mg_matrix_N(nx,ny,n_level;v1=2,v2=2,v3=2,tolerance=1e-10)
    maximum_iterations = 10000 # set maximum_iterations
    u_n, f_array = initialize_uf(nx,ny)
    dx = 1.0 ./nx
    dy = 1.0 ./ny
    poisson_matrix_ = poisson_matrix(nx,ny,dx,dy)
    u_mg = Matrix{Float64}[]
    f_mg = Matrix{Float64}[]
    A_mg = SparseMatrixCSC{Float64, Int64}[]
    L_mg = LowerTriangular{Float64, SparseMatrixCSC{Float64, Int64}}[]
    U_mg = UpperTriangular{Float64, SparseMatrixCSC{Float64, Int64}}[]
    rest_mg = SparseMatrixCSC{Float64, Int64}[]
    prol_mg = SparseMatrixCSC{Float64, Int64}[]
    r = zeros(Float64,nx+1, ny+1)

    push!(u_mg, u_n)
    push!(f_mg, f_array)
    push!(A_mg, poisson_matrix_)
    L = LowerTriangular(poisson_matrix_)
    # U = poisson_matrix_ - L # create dense matrix
    # U = copy(UpperTriangular(poisson_matrix_)) #Can not directly change the UpperTriangular(poisson_matrix_)
    # for i in 1:size(U)[1]
    #     U[i,i] = 0
    # end
    U = triu(poisson_matrix_, 1)
    push!(L_mg, L)
    push!(U_mg, U)

    # compute initial residual

    r[:] = f_array[:] - A_mg[1]*u_n[:]


    # Compute initial L-2 norm
    rms = compute_l2norm(nx,ny,r)

    init_rms = rms
    mg_iter_count = 0

    println("0", " ", rms, " ", rms/init_rms)

    if nx < (2^n_level)
        print("Number of levels exceeds the possible nmber.\n")
    end

    # allocate memory for grid size at different levels

    lnx = zeros(Int64, n_level)
    lny = zeros(Int64, n_level)
    ldx = zeros(Float64, n_level)
    ldy = zeros(Float64, n_level)

    # initialize the mesh details at fine level
    lnx[1] = nx
    lny[1] = ny
    ldx[1] = dx
    ldy[1] = dy

    # calclate mesh details for coarse levels and allocate matrix
    # numerical solution and error restricted from upper level

    for i in 2:n_level
        lnx[i] = Int64(lnx[i-1]/2)
        lny[i] = Int64(lny[i-1]/2)
        ldx[i] = ldx[i-1]*2
        ldy[i] = ldy[i-1]*2

        # allocate matrix for storage at coarse levels
        fc = zeros(Float64, lnx[i]+1, lny[i]+1)
        unc = zeros(Float64, lnx[i]+1, lny[i]+1)

        push!(u_mg, unc)
        push!(f_mg, fc)
    end

    println("Starting assembling restriction matrices")
    for k in 1:n_level-1
        push!(rest_mg, restriction_matrix(lnx[k],lny[k],lnx[k+1],lny[k+1]))
        push!(prol_mg, prolongation_matrix(lnx[k],lny[k],lnx[k+1],lny[k+1]))
    end
    println("Finishing assembling restriction matrices")

    # allocate matrix for storage at fine level
    # residual at fine level is already defined at global level

    prol_fine = zeros(Float64, lnx[1]+1, lny[1]+1)

    # temporary residual which is restricted to coarse mesh error
    # the size keeps on changing

    temp_residual = zeros(Float64, lnx[1]+1, lny[1]+1)

    # u_n .= reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
    # reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

    println("Starting Multigrid Iterations")
    for iteration_count = 1:maximum_iterations
        mg_iter_count += 1
        for i in 1:v1
            u_mg[1] .= reshape(L_mg[1]\(f_mg[1][:] .- U_mg[1]*u_mg[1][:]), nx+1, ny+1)
        end

        # calculate residual
        r = reshape((f_array[:] .- poisson_matrix_ * u_mg[1][:]),nx+1,ny+1)

        # compute l2norm of the residual
        rms = compute_l2norm(lnx[1],lny[1],r)

        # write results only for the finest residual
        # ...

        # count = iteration_count
        # count = iteration_count
        # @show (rms)
        println(mg_iter_count, " ", rms, " ", rms/init_rms)
        if (rms/init_rms) <= tolerance
            break
        end

        # from second level to coarsest level
        for k = 2:n_level
            if k == 2
                # for second level temporary residual is taken from fine mesh level
                temp_residual = r
            else
                # from third level onwards residual is computed for (k-1) level
                # which will be restricted to kth level error
                temp_residual = zeros(Float64, lnx[k-1]+1, lny[k-1]+1)
                compute_residual(lnx[k-1], lny[k-1], ldx[k-1], ldy[k-1],
                            f_mg[k-1], u_mg[k-1], temp_residual)
            end
            # restriction(lnx[k-1], lny[k-1], lnx[k], lny[k], temp_residual,
            #                 f_mg[k])
            # f_mg[k][:] ≈ restriction_matrix(lnx[k-1],lny[k-1],lnx[k],lny[k]) * temp_residual[:]
            # @show k
            # f_mg[k][:] = restriction_matrix(lnx[k-1],lny[k-1],lnx[k],lny[k]) * temp_residual[:]
            f_mg[k][:] = rest_mg[k-1] * temp_residual[:]


            # solution at kth level to zero
            u_mg[k][:,:] = zeros(lnx[k]+1, lny[k]+1)
            
            # formulating Poisson matrix
            if length(A_mg) < k # pushing A_mg L_mg U_mg if they are not formulated
                println("Assembling matrices for Nx = $(lnx[k]), Ny = $(lny[k]) for the first time")
                push!(A_mg,poisson_matrix(lnx[k],lny[k],ldx[k],ldy[k]))
                push!(L_mg, LowerTriangular(A_mg[k]))
                # U = copy(UpperTriangular(A_mg[k])) #Can not directly change the UpperTriangular(poisson_matrix_)
                # for i in 1:size(U)[1]
                #     U[i,i] = 0
                # end
                U = triu(A_mg[k],1)
                push!(U_mg, U)
            end

            # solve (∇^-λ^2)ϕ = ϵ on coarse grid (kthe level)
            if k < n_level
                for i in 1:v1
                    u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] .- U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1)
                end
            elseif k == n_level
                for i in 1:v2
                    u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] .- U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1)
                end
            end
        end


        # sweep from coarsest grid to finest grid
        for k = n_level:-1:2
            # temporary matrix for correction storage at the (k-1)th level
            # solution prolongated from the kth level to the (k-1)th level
            prol_fine = zeros(Float64, lnx[k-1]+1, lny[k-1]+1)

            # prolongate solution from (k)th level to (k-1)th level
            # prol_fine[:] = prolongation_matrix(lnx[k-1],lny[k-1],lnx[k],lny[k]) * u_mg[k][:]
            prol_fine[:] = prol_mg[k-1] * u_mg[k][:]

            # update u_mg

            for j = 2:lnx[k-1] for i = 2:lny[k-1]
                u_mg[k-1][i,j] = u_mg[k-1][i,j] + prol_fine[i,j]
            end end

            # Gauss seidel iteration
            for i in 1:v3
                u_mg[k-1] .= reshape(L_mg[k-1]\(f_mg[k-1][:] .- U_mg[k-1]*u_mg[k-1][:]),lnx[k-1]+1,lny[k-1]+1)
            end
        end
    end
end
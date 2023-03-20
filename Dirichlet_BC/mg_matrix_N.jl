using CPUTime
using Printf
using Plots
using SparseArrays


function compute_residual(nx,ny,dx,dy,f,u_n,r)
    for j in 2:ny for i = 2:nx
        d2udx2 = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx^2)
        d2udy2 = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy^2)
        r[i,j] = f[i,j]  - d2udx2 - d2udy2
        # r[i,j] = d2udx2 + d2udy2
    end end
end


function poisson_matrix(nx,ny,dx,dy)
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
    for j in 1:ny+ 1for i in 1:nx + 1
        index = (j - 1) * (nx+1) + i
        if (i == 1 || i == nx + 1 || j == 1 || j == ny + 1)
            poisson_matrix_[index,:] .= 0
            poisson_matrix_[index,index] = 1
            index_count += 1
        end
    end end
    @show index_count
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



function restriction(nxf, nyf, nxc, nyc, r, ec)

    for j = 2:nyc for i = 2:nxc
        # grid index for fine grid for the same coarse point
        center = 4.0*r[2*i-1, 2*j-1]
        # E, W, N, S with respect to coarse grid point in fine grid
        grid = 2.0*(r[2*i-1, 2*j-1+1] + r[2*i-1, 2*j-1-1] +
                    r[2*i-1+1, 2*j-1] + r[2*i-1-1, 2*j-1])
        # NE, NW, SE, SW with respect to coarse grid point in fine grid
        corner = 1.0*(r[2*i-1+1, 2*j-1+1] + r[2*i-1+1, 2*j-1-1] +
                      r[2*i-1-1, 2*j-1+1] + r[2*i-1-1, 2*j-1-1])
        # restriction using trapezoidal rule
        ec[i,j] = (center + grid + corner)/16.0
    end end

    # restriction for boundary points bottom and top
    for j = 1:nxc+1
        # bottom boundary i = 1
        ec[1,j] = r[1, 2*j-1]
        # top boundary i = ny_coarse+1
        ec[nyc+1,j] = r[nyf+1, 2*j-1]
    end

    # restriction for boundary poinys left and right
    for i = 1:nyc+1
        # left boundary j = 1
        ec[i,1] = r[2*i-1,1]
        # right boundary nx_coarse+1
        ec[i,nxc+1] = r[2*i-1, nyf+1]
    end
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


function prolongation(nxc, nyc, nxf, nyf, unc, ef)
    for j = 1:nyc for i = 1:nxc
        # direct injection at center point
        ef[2*i-1, 2*j-1] = unc[i,j]
        # east neighnour on fine grid corresponding to coarse grid point
        ef[2*i-1, 2*j-1+1] = 0.5*(unc[i,j] + unc[i,j+1])
        # north neighbout on fine grid corresponding to coarse grid point
        ef[2*i-1+1, 2*j-1] = 0.5*(unc[i,j] + unc[i+1,j])
        # NE neighbour on fine grid corresponding to coarse grid point
        ef[2*i-1+1, 2*j-1+1] = 0.25*(unc[i,j] + unc[i,j+1] +
                                     unc[i+1,j] + unc[i+1,j+1])
    end end

    # update boundary points
    for i = 1:nyc+1
        # left boundary j = 1
        ef[2*i-1,1] = unc[i,1]
        # right boundary j = nx_fine+1
        ef[2*i-1, nyf+1] = unc[i,nxc+1]
    end

    for j = 1:nxc+1
        #bottom boundary i = 1
        ef[1,2*j-1] = unc[1,j]
        # top boundary i =  ny_fine+1
        ef[nyf+1,2*j-1] = unc[nyc+1,j]
    end
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
function prolongation_matrix(nxc,nyc,nxf,nyf)
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



#------------------------------------------------------------------------------
ipr = 1

nx = ny = Int64(64)
n_level = 3

tolerance = Float64(1.0e-10)

x_l = 0.0
x_r = 1.0
y_b = 0.0
y_t = 1.0

v1 = 2 # relaxation
v2 = 2 # prolongation
v3 = 2 # coarsest level

dx = (x_r - x_l) / nx
dy = (y_t - y_b) / ny


x = Array{Float64}(undef, nx+1)
y = Array{Float64}(undef, ny+1)
u_e = Array{Float64}(undef, nx+1, ny+1)

f_array = Array{Float64}(undef, nx+1, ny+1)
r_array = spzeros(nx+1, ny+1)
u_n = Array{Float64}(undef, nx+1, ny+1)

for i = 1:nx+1
    x[i] = x_l + dx*(i-1)
end
for i = 1:ny+1
    y[i] = y_b + dy*(i-1)
end

c1 = (1.0/16.0)^2
c2 = -2.0*pi*pi

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


function reset_uf(u_n, f_array)
    # u_n = Array{Float64}(undef, nx+1, ny+1)

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
end


# modify f_array to match BC boundary conditions
for i = 1:nx+1 for j = 1:ny+1
    if ((i == 1) || (i == nx+1) || (j == 1) || (j == ny+1))
        f_array[i,j] = u_n[i,j]
    end
end end

compute_residual(nx,ny,dx,dy,f_array,u_n,r_array)

poisson_matrix_ = poisson_matrix(nx,ny,dx,dy)
poisson_u_n = reshape(poisson_matrix_ * u_n[:], (nx+1), (nx+1))

# @show (r_array[:] - poisson_matrix_ * u_n[:])

# r_array - reshape(f_array[:] - poisson_matrix_ * u_n[:],nx+1,ny+1)

manual_residual = reshape((f_array[:] - poisson_matrix_ * u_n[:]),nx+1,ny+1)
@assert manual_residual ≈ r_array

V = 1

f_array

u_n

u_n_copy = copy(u_n)

gauss_seidel_mg(nx,ny,dx,dy,f_array,u_n_copy, V)


# u_n + (f_array - reshape(poisson_matrix_*u_n[:],nx+1,ny+1)) ./ (-2.0/dx^2 -2.0/dy^2)

L = LowerTriangular(poisson_matrix_)
U = poisson_matrix_ - L
u_n_new = reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

@assert u_n_copy ≈ u_n_new
# reshape(u_n[:] + L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

#######################################################################
## Starting multigrid

u_mg = Matrix{Float64}[]
f_mg = Matrix{Float64}[]
A_mg = Matrix{Float64}[]
L_mg = Matrix{Float64}[]
U_mg = Matrix{Float64}[]
r = zeros(Float64,nx+1, ny+1)

push!(u_mg, u_n)
push!(f_mg, f_array)
push!(A_mg, poisson_matrix_)
push!(L_mg, L)
push!(U_mg, U)

# compute initial residual

r[:] = f_array[:] - poisson_matrix_ * u_n[:]


# Compute initial L-2 norm
rms = compute_l2norm(nx,ny,r)

init_rms = rms

print("0", " ", rms, " ", rms/init_rms)

if nx < (2^n_level)
    print("Number of levels exceeds the possible nmber.\n")
end

# allocate memory for grid size at different levels

lnx = zeros(Int64, n_level)
lnx = zeros(Int64, n_level)
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

# allocate matrix for storage at fine level
# residual at fine level is already defined at global level

pro_fine = zeros(Float64, lnx[1]+1, lny[1]+1)

# temporary residual which is restricted to coarse mesh error
# the size keeps on changing

temp_residual = zeros(Float64, lnx[1]+1, lny[1]+1)

# u_n .= reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
# reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

maximum_iterations = 10
# for iteration_count = 1:maximum_iterations
    for i in 1:v1
        u_mg[1] .= reshape(L\(f_array[:] - U*u_mg[1][:]), nx+1, ny+1)
    end

    # calculate residual
    r = reshape((f_array[:] - poisson_matrix_ * u_mg[1][:]),nx+1,ny+1)

    # compute l2norm of the residual
    rms = compute_l2norm(lnx[1],lny[1],r)

    # write results only for the finest residual
    # ...

    # count = iteration_count
    # count = iteration_count


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
        f_mg[k][:] = restriction_matrix(lnx[k-1],lny[k-1],lnx[k],lny[k]) * temp_residual[:]

        # solution at kth level to zero
        u_mg[k][:,:] = zeros(lnx[k]+1, lny[k]+1)
        
        # formulating Poisson matrix
        if length(A_mg) < k # pushing A_mg L_mg U_mg if they are not formulated
            push!(A_mg,poisson_matrix(lnx[k],lny[k],ldx[k],ldy[k]))
            push!(L_mg, LowerTriangular(A_mg[k]))
            push!(U_mg, A_mg[k] - L_mg[k])
        end

        # solve (∇^-λ^2)ϕ = ϵ on coarse grid (kthe level)
        if k < n_level
            for i in 1:v1
                u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] - U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1)
            end
        elseif k == n_level
            for i in 1:v2
                u_mg[k] .= reshape(L_mg[k]\(f_mg[k][:] - U_mg[k]*u_mg[k][:]),lnx[k]+1,lny[k]+1)
            end
        end

        for k = n_level:-1:2
            # temporary matrix for correction storage at the (k-1)th level
            # solution prolongated from the kth level to the (k-1)th level
            prol_fine = zeros(Float64, lnx[k-1]+1, lny[k-1]+1)

            # prolongate solution from (k)th level to (k-1)th level
            prol_fine[:] = prolongation_matrix(lnx[k],lny[k],lnx[k-1],lny[k-1]) * u_mg[k][:]

            # update u_mg

            for j = 2:lnx[k-1] for i = 2:lny[k-1]
                u_mg[k-1][i,j] = u_mg[k-1][i,j] + prol_fine[i,j]
            end end

            # Gauss seidel iteration
            for i in 1:v3
                u_mg[k-1] .= reshape(L_mg[k-1]\(f_mg[k-1][:] - U_mg[k-1]*u_mg[k-1][:]),lnx[k-1]+1,lny[k-1]+1)
            end

        end

    end

# end
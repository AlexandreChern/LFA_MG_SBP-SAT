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
function prolongation_matrix(nxf,nyf,nxc,nyc)
    prolongation_matrix_ = spzeros((nxf+1)*(nyf+1),(nxc+1)*(nyc+1))
    for j in 1:nyc+1
        for i in 1:nxc+1
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            # @show (indexc, indexf)
            prolongation_matrix_[indexf,indexc] = (4)/16.0
            if 2 <= i <= nxc
                prolongation_matrix_[indexf+1,indexc] = (2)/16.0
                prolongation_matrix_[indexf-1,indexc] = (2)/16.0
            end
            if 2 <= j <= nxc
                prolongation_matrix_[indexf+nyf+1,indexc] = (2)/16.0
                prolongation_matrix_[indexf-nyf-1,indexc] = (2)/16.0
            end
            if (2 <= i <= nxc) && (2 <= j <= nyc)
                prolongation_matrix_[indexf-nyf-1-1,indexc] = (1)/16.0
                prolongation_matrix_[indexf-nyf-1+1,indexc] = (1)/16.0
                prolongation_matrix_[indexf+nyf+1+1,indexc] = (1)/16.0
                prolongation_matrix_[indexf+nyf+1-1,indexc] = (1)/16.0
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


compute_residual(nx,ny,dx,dy,f_array,u_n,r_array)

poisson_matrix_ = poisson_matrix(nx,ny,dx,dy)
poisson_u_n = reshape(poisson_matrix_ * u_n[:], (nx+1), (nx+1))

@show (r_array[:] - poisson_matrix_ * u_n[:])

manual_residual = reshape((f_array[:] - poisson_matrix_ * u_n[:]),nx+1,ny+1)

V = 1

f_array

u_n

u_n_copy = copy(u_n)

gauss_seidel_mg(nx,ny,dx,dy,f_array,u_n_copy, V)


# u_n + (f_array - reshape(poisson_matrix_*u_n[:],nx+1,ny+1)) ./ (-2.0/dx^2 -2.0/dy^2)

L = LowerTriangular(poisson_matrix_)
U = poisson_matrix_ - L

reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
# reshape(u_n[:] + L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
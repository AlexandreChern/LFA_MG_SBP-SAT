include("mg_matrix_N.jl")
include("mg_N.jl")


#------------------------------------------------------------------------------


ipr = 1

nx = ny = Int64(256)
n_level = 4

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
# U = poisson_matrix_ - L
U = copy(UpperTriangular(poisson_matrix_)) #Can not directly change the UpperTriangular(poisson_matrix_)

for i in 1:size(U)[1]
    U[i,i] = 0
end
u_n_new = reshape(L\(f_array[:] - U*u_n[:]), nx+1, ny+1)

@assert u_n_copy ≈ u_n_new
# reshape(u_n[:] + L\(f_array[:] - U*u_n[:]), nx+1, ny+1)
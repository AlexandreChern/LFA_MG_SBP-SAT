using SparseArrays
using LinearAlgebra

"""
    prolongation_matrix_v2(nxf,nyf,nxc,nyc)
    Generating SBP-preserving restriction matrix from coarse grid (nxc, nyc) to fine grid (nxf,nyf)
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
function prolongation_matrix_v2(nxf,nyf,nxc,nyc) 
    # SBP preserving prolongation matrix
    prolongation_matrix_x = spzeros(nxf+1,nxc+1)
    for i in 1:nxc
        prolongation_matrix_x[2*i-1,i] = 1
        prolongation_matrix_x[2*i,i] = 0.5
        prolongation_matrix_x[2*i,i+1] = 0.5
    end
    prolongation_matrix_x[end,end] = 1

    prolongation_matrix_y = spzeros(nyf+1,nyc+1)
    for i in 1:nyc
        prolongation_matrix_y[2*i-1,i] = 1
        prolongation_matrix_y[2*i,i] = 0.5
        prolongation_matrix_y[2*i,i+1] = 0.5
    end
    prolongation_matrix_y[end,end] = 1

    prolongation_matrix_ = kron(prolongation_matrix_x,prolongation_matrix_y)
    return prolongation_matrix_
end




"""
    restriction_matrix_v2(nxf,nyf,nxc,nyc)
    Generating SBP-preserving restriction matrix from fine grid (nxf,nyf) to coarse grid (nxc, nyc)
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
function restriction_matrix_v2(nxf,nyf,nxc,nyc)
    restriction_matrix_x = spzeros(nxc+1, nxf+1)
    restriction_matrix_x[1,1] = 1/2
    restriction_matrix_x[1,2] = 1/2
    for i in 2:nxc
        restriction_matrix_x[i,2*i-1] = 1/2
        restriction_matrix_x[i,2*i-2] = 1/4
        restriction_matrix_x[i,2*i] = 1/4
    end
    restriction_matrix_x[end,end] = 1/2
    restriction_matrix_x[end,end-1] = 1/2

    restriction_matrix_y = spzeros(nyc+1, nyf+1)
    restriction_matrix_y[1,1] = 1/2
    restriction_matrix_y[1,2] = 1/2
    for i in 2:nyc
        restriction_matrix_y[i,2*i-1] = 1/2
        restriction_matrix_y[i,2*i-2] = 1/4
        restriction_matrix_y[i,2*i] = 1/4
    end
    restriction_matrix_y[end,end] = 1/2
    restriction_matrix_y[end,end-1] = 1/2

    restriction_matrix_ = kron(restriction_matrix_x,restriction_matrix_y)
    return restriction_matrix_
end


"""
    prolongation_matrix_v3(nxf,nyf,nxc,nyc)
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
function prolongation_matrix_v3(nxf,nyf,nxc,nyc)
    Ix = spdiagm(-1 => fill(-1.0,nxf-1), 0 => fill(2.0,nxf), 1 => fill(-1.0,nxf-1))
    Iy = spdiagm(-1 => fill(-1.0,nyf-1), 0 => fill(2.0,nyf), 1 => fill(-1.0,nyf-1))
    P = kron(Iy,Ix)
    P = P[1:end-nyf,1:end-1]
    P = [P[:,1] P P[:,end]]
    # P = kron(speye(nyc+1),kron(speye(nxc+1),P))
    P = kron(sparse(I,nyc+1,nyc+1),kron(sparse(I,nxc+1,nxc+1),P))
    return P
end
using SparseArrays
using LinearAlgebra




"""
    restriction_matrix_v0(nxf,nyf,nxc,nyc)
    Generating standard restriction matrix from fine grid (nxf,nyf) to coarse grid (nxc, nyc)
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
function restriction_matrix_v0(nxf,nyf,nxc,nyc)
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
    prolongation_matrix_v0(nxf,nyf,nxc,nyc)
    Generating standard prolongation matrix from coarse grid (nxc, nyc) to fine grid (nxf,nyf)
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
function prolongation_matrix_v0(nxf,nyf,nxc,nyc) 
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
    prolongation_matrix_v2(nxf,nyf,nxc,nyc)
    Generating SBP-preserving prolongation matrix from coarse grid (nxc, nyc) to fine grid (nxf,nyf)
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
    # Ix = spdiagm(-1 => fill(-1.0,nxf-1), 0 => fill(2.0,nxf), 1 => fill(-1.0,nxf-1))
    # Iy = spdiagm(-1 => fill(-1.0,nyf-1), 0 => fill(2.0,nyf), 1 => fill(-1.0,nyf-1))
    # P = kron(Iy,Ix)
    # P = P[1:end-nyf,1:end-1]
    # P = [P[:,1] P P[:,end]]
    # # P = kron(speye(nyc+1),kron(speye(nxc+1),P))
    # P = kron(sparse(I,nyc+1,nyc+1),kron(sparse(I,nxc+1,nxc+1),P))
    # return P
    # prolongation_matrix_x = spzeros(nxf+1,nxc+1)
    # for i in 1:nxc
    #     prolongation_matrix_x[2*i-1,i] = 1
    #     prolongation_matrix_x[2*i,i] = 0.5
    #     prolongation_matrix_x[2*i,i+1] = 0.5
    # end
    # prolongation_matrix_x[end,end] = 1

    # prolongation_matrix_y = spzeros(nyf+1,nyc+1)
    # for i in 1:nyc
    #     prolongation_matrix_y[2*i-1,i] = 1
    #     prolongation_matrix_y[2*i,i] = 0.5
    #     prolongation_matrix_y[2*i,i+1] = 0.5
    # end
    # prolongation_matrix_y[end,end] = 1

    # prolongation_matrix_ = kron(prolongation_matrix_x,prolongation_matrix_y)
    # return prolongation_matrix_
    prolongation_matrix_ = spzeros((nxf+1)*(nyf+1),(nxc+1)*(nyc+1))
    for j in 1:nyc+1
        for i in 1:nxc+1
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            # @show (indexc, indexf)
            prolongation_matrix_[indexf,indexc] = 1.0  # direct injection instead of(4)/16.0
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
function prolongation_matrix_v1(nxf,nyf,nxc,nyc)
    prolongation_matrix_ = spzeros((nxf+1)*(nyf+1),(nxc+1)*(nyc+1))
    for j in 1:nyc
        for i in 1:nxc
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            # @show (indexc, indexf)
            prolongation_matrix_[indexf,indexc] = 1.0  # direct injection instead of(4)/16.0

            # east neighbors
            prolongation_matrix_[indexf + nyf + 1, indexc] = 0.5
            prolongation_matrix_[indexf + nyf + 1, indexc + nxc + 1] = 0.5

            # north neighbors
            prolongation_matrix_[indexf + 1, indexc] = 0.5
            prolongation_matrix_[indexf + 1, indexc + 1] = 0.5

            # prolongation_matrix_[indexf+1,indexc] = 0.5 #(2)/16.0
            # prolongation_matrix_[indexf-1,indexc] = 0.5 #(2)/16.0
            
            # prolongation_matrix_[indexf+nyf+1,indexc] = 0.5 #(2)/16.0
            # prolongation_matrix_[indexf-nyf-1,indexc] = 0.5 #(2)/16.0
            
            # north east neighbors
            # prolongation_matrix_[indexf-nyf-1-1,indexc] = 0.25 #(1)/16.0
            # prolongation_matrix_[indexf-nyf-1+1,indexc] = 0.25 #(1)/16.0
            # prolongation_matrix_[indexf+nyf+1+1,indexc] = 0.25 #(1)/16.0
            # prolongation_matrix_[indexf+nyf+1-1,indexc] = 0.25 #(1)/16.0
            prolongation_matrix_[indexf + nyf + 1 + 1, indexc] = 0.25
            prolongation_matrix_[indexf + nyf + 1 + 1, indexc + nxc + 1] = 0.25
            prolongation_matrix_[indexf + nyf + 1 + 1, indexc + 1] = 0.25
            prolongation_matrix_[indexf + nyf + 1 + 1, indexc + nxc + 1 + 1] = 0.25
        end
    end

    for i = 1:nyc+1
        # left boundary
        prolongation_matrix_[(2*i-1),i] = 1
        # right boundary
        prolongation_matrix_[(2*i-1 + nyf * (nxf+1) ), i + nyc * (nxc+1)] = 1
    end

    for j = 1:nxc+1
        # left boundary
        prolongation_matrix_[1 + (2*j-1 - 1) * (nxf+1), 1 + (j-1) * (nxc+1)] = 1
        # right boundary
        prolongation_matrix_[nxf+1 + (2*j-1 - 1) * (nxf+1), nxc+1 + (j-1) * (nxc+1)] = 1
    end

    return prolongation_matrix_
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
function restriction_matrix_v1(nxf,nyf,nxc,nyc)
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

    for j in 1:nyc+1
        for i in 1:nyc+1
            indexc = (j-1)* (nxc+1) + i
            indexf = (2*j-1 -1) * (nxf+1) + (2*i-1) # careful about this index
            if i == 1 || i == nxc+1 || j == 1 || j == nyc+1
                restriction_matrix_[indexc,:] .= 0
                restriction_matrix_[indexc,indexf] = 1
            end
        end
    end
    return restriction_matrix_
end


function operator_dependent_restriction(input_A)
    nx =  ny = Int(sqrt(size(input_A,1)))
    indices = ((Vector(1:2:nx) .- 1) * nx .+ Vector(1:2:ny)')'[:]
    tmp = input_A[indices,:]
    row = Int(1)
    for idx in indices
        tmp[row,:] ./= -tmp[row,idx]
        tmp[row,idx] = 0
        row += Int(1)
    end
    return tmp
end

# indices = ((Vector(1:2:9) .- 1) * 9 .+ Vector(1:2:9)')'[:]
# mg_struct.A_mg[2][indices,:]
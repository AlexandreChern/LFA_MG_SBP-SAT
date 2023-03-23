using SparseArrays
using LinearAlgebra

"""
    prolongation_matrix_v2(nxf,nyf,nxc,nyc)
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
function prolongation_matrix_v2(nxf,nyf,nxc,nyc)
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




"""
    restriction_matrix_v2(nxf,nyf,nxc,nyc)
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
function restriction_matrix_v2(nxf,nyf,nxc,nyc)
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
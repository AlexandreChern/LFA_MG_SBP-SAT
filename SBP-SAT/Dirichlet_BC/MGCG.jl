mutable struct MG
    A_mg
    L_mg
    U_mg
    f_mg
    u_mg
end

mg_struct = MG([],[],[],[],[])

function initialze_mg_struct(mg_struct)
end

function mgcg(mg_struct;Nx=64,Ny=64,n_levels=3)
    A_mg = mg_struct.A_mg
    f_mg = mg_struct.f_mg
    u_mg = mg_struct.u_mg
    if isempty(A_mg)
        # Assembling matrices
    end
end
function m = mass(P, mu)
%MASS Diagonal omega^2/c^2 = omega^2*(1 + mu*g) of the waveguide operator.
    m = P.mass_base + mu * P.mass_parameter;
end

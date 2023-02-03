import numpy as np

def getAcc( pos, mass, G, softening):
    
    """
    pos is an N x 3 matrix of positions (3 spatial Dimensions for each N)
    mass is a N x 1 vector of masses (1 Mass dimension for each N)
    G is Newtons Gravitational Constant 
    softening is a small constant to account for the point-particles acceleration under INV Sq Law

    returns a N x 3 matrix of acceleration (3 spatial dimensions for each N)
 
    Newtowns law of acceleration
    a[i] = G * SIGMA m[j] * ( r[j] - r[i] ) / ABS (r[j] - r[i] ^3)
    acceleration_of_body_i = grav_constant * sum_of( distance_between_i_and_j / cube_of( abs_val(distance_between_and_i_j))
    """

    # Create a column vector for the specified dimensional values for each node
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # bizaaro numpy broadcasting used to create a pairwise matrix of (pos_d_i - pos_d_j) where d is the unit dimension x,y,z
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # 1/r^3 for all pairwaise particle seperations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-3/2)

    # determine the acceleration componant for each node along each dimension
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # Pack each acceleration vector and return the matrix
    return np.hstack((ax, ay, az))




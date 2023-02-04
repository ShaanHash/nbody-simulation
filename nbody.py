import numpy as np
import matplotlib.pyplot as plt

def getAcceleration( position, mass, G, softening):
    
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
    x_positions = position[:, 0:1]
    y_positions = position[:, 1:2]
    z_positions = position[:, 2:3]

    # bizaaro numpy broadcasting used to create a pairwise matrix of (pos_d_i - pos_d_j) where d is the unit dimension x,y,z
    delta_x = x_positions.T - x_positions
    delta_y = y_positions.T - y_positions
    delta_z = z_positions.T - z_positions

    # 1/r^3 for all pairwaise particle seperations
    inverse_r3 = (delta_x**2 + delta_y**2 + delta_z**2 + softening**2)**(-3/2)

    # determine the acceleration componant for each node along each dimension
    ax = G * (delta_x * inverse_r3) @ mass
    ay = G * (delta_y * inverse_r3) @ mass
    az = G * (delta_z * inverse_r3) @ mass

    # Pack each acceleration vector and return the matrix
    return np.hstack((ax, ay, az))


def main():

    # Intial Simulation State
    number_of_particles = 10
    current_time = 0
    end_time = 10
    time_step = 0.01
    softening = 0.01
    grav_constant = 1
    real_time_plotting = True
    
    # Find the integer number of time steps in the simulation
    number_of_time_steps = int(
        np.ceil(
            end_time / time_step
            ))

    # Starting particle properties
    mass = 20.0 * np.ones((number_of_particles, 1)) / number_of_particles
    positions = np.random.randn(number_of_particles, 3)
    velocities = np.random.randn(number_of_particles, 3)

    # Starting accelerations
    accelerations = getAcceleration(positions, mass, grav_constant, softening)

    # Some wierd physics to convert to center of mass inertial frame
    velocities -= np.mean(mass * velocities, 0) / np.mean(mass)

    # Create a figure
    figure = plt.figure(figsize=(4,5), dpi=80)
    # Define a grid spec so we can place our subplots
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    # Create the first plot and axis for the nbody simulation
    ax1 = plt.subplot(grid[0:2,0])
    # Create a second plot to track the KinEnergy and PotEnergy
    ax2 = plt.subplot(grid[2,0])


    # Step through each time interval
    for i in range(number_of_time_steps):

        # Half step kick
        velocities += accelerations * time_step/2.0

        # drift
        positions += velocities * time_step

        # Update accelerations
        accelerations = getAcceleration(positions, mass, grav_constant, softening)

        # Step foward in time
        current_time += time_step

        # If were real-time plotting or after time_step -1 intervals; plot the figure
        if real_time_plotting or (i == time_step-1):
            plt.sca(ax1)
            plt.cla()
            plt.scatter(positions[:,0],positions[:,1],s=20,color='blue')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])
			
            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, end_time), ylim=(-300, 300))
            ax2.set_aspect(0.007)
			
            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    ax2.legend(loc='upper right')

    # Save figure
    plt.savefig('nbody.png',dpi=240)
    plt.show()

    return 0
	    
if __name__ == "__main__":
    main()


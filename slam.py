from __future__ import division
import numpy as np
import slam_utils
import tree_extraction

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.
        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    ve = u[0]
    alpha = u[1]
    phi = ekf_state['x'][2]

    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']
    vc = ve / (1 - (np.tan(alpha)*(H/L)))

    motion = np.zeros((3,))
    motion[0] = dt*vc*(np.cos(phi) - ((1/L)*np.tan(alpha)*(a*np.sin(phi) + b*np.cos(phi))))
    motion[1] = dt*vc*(np.sin(phi) + ((1/L)*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi))))
    motion[2] = dt*((vc/L)*np.tan(alpha))

    G = np.eye(3)
    G[0,2] = dt*(-vc*np.sin(phi) - ((vc/L)*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi))))
    G[1,2] = dt*(vc*np.cos(phi) + ((vc/L)*np.tan(alpha)*(-a*np.sin(phi) - b*np.cos(phi))))

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    x_var = sigmas['xy']
    y_var = sigmas['xy']
    phi_var = sigmas['phi']
    sigma_prev = ekf_state['P'][0:3,0:3]
    motion,G = motion_model(u, dt, ekf_state, vehicle_params)

    ekf_state['x'][0:3] += motion
    R_t = np.diag([x_var**2, y_var**2, phi_var**2])
    sigma = slam_utils.make_symmetric(np.matmul(G,np.matmul(sigma_prev, G.T)) + R_t)
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    ekf_state['P'][0:3,0:3] = sigma

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''

    H = np.zeros((2,3))
    H[0,0] = 1
    H[1,1] = 1
    sigma = ekf_state['P']
    Q_t = np.diag([sigmas['gps']**2, sigmas['gps']**2])

    S = np.linalg.inv(np.matmul(np.matmul(H, sigma[0:3, 0:3]), H.T) + Q_t)
    r = np.reshape(gps - ekf_state['x'][0:2], (-1,1))
    d = np.matmul(np.matmul(r.T, S), r)

    if (d > 13.8):
        return ekf_state

    K_last_term = S
    K = np.matmul(np.matmul(sigma[0:3,0:3], H.T), K_last_term)
    ekf_state['x'][0:3] += np.matmul(K, r)[:,0]
    ekf_state['P'][0:3,0:3] = slam_utils.make_symmetric(np.matmul((np.eye(3) - np.matmul(K, H)), sigma[0:3,0:3]))

    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''

    n = ekf_state['x'].shape[0]

    xl = ekf_state['x'][2*landmark_id + 3]
    yl = ekf_state['x'][2*landmark_id + 4]
    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]

    zhat = np.zeros((2,1))
    zhat[0,0] = np.sqrt(pow(xl - xv, 2) + pow(yl - yv, 2))
    zhat[1,0] = slam_utils.clamp_angle(np.arctan2((yl - yv),(xl - xv)) - phi)

    denom = np.sqrt(pow(xl - xv, 2) + pow(yl - yv, 2))
    H = np.zeros((2,n))

    landmark = np.zeros((2,2))
    landmark[0,0] = (xl - xv)/denom
    landmark[0,1] = (yl - yv)/denom
    landmark[1,0] = -(yl - yv)/pow(denom,2)
    landmark[1,1] = (xl - xv)/pow(denom,2)

    H[0,0] = -(xl - xv)/denom
    H[0,1] = -(yl - yv)/denom
    H[1,0] = (yl - yv)/pow(denom,2)
    H[1,1] = -(xl - xv)/pow(denom,2)
    H[1,2] = -1

    H[:,(2*landmark_id + 3):(2*landmark_id + 5)] = landmark

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    range = tree[0]
    angle = tree[1]

    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]

    xl = range * np.cos(slam_utils.clamp_angle(angle + phi)) + xv
    yl = range * np.sin(slam_utils.clamp_angle(angle + phi)) + yv

    ekf_state['x'] = np.append(ekf_state['x'], [xl,yl])

    m,n = ekf_state['P'].shape
    P_new = np.zeros((m+2,n+2))
    P_new[m,n] = 69.5
    P_new[m+1,n+1] = 69.5
    P_new[0:m,0:n] = ekf_state['P']
    ekf_state['P'] = slam_utils.make_symmetric(P_new)

    ekf_state['num_landmarks'] += 1

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    m = len(measurements)
    n = ekf_state['num_landmarks']

    if m > n:
        M = np.full((m,n+m), 6.1)
    else:
        M = np.zeros((m,n))

    Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])
    sigma = ekf_state['P']

    for i in range(m):
        z = np.zeros((2,1))
        z[0,0] = measurements[i][0]
        z[1,0] = measurements[i][1]
        for j in range(n):
            zhat,H = laser_measurement_model(ekf_state, j)
            r = z - zhat
            S = np.matmul(H, np.matmul(sigma, H.T)) + Q
            M[i,j] = np.matmul(r.T, np.matmul(np.linalg.inv(S), r))

    pairs = slam_utils.solve_cost_matrix_heuristic(np.copy(M))
    assoc = [0] * len(measurements)

    for i,j in pairs:
        if j >= n:
            min_element = np.amin(M[i,0:n-1])
            #13.8
            if (min_element > 18.2): #chi(0.9999)
                assoc[i] = -1
            else:
                assoc[i] = -2
        else:
            if M[i,j] > 6:
                if M[i,j] > 18.2:
                    assoc[i] = -1
                else:
                    assoc[i] = -2
            else:
                assoc[i] = j

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    for i,j in enumerate(assoc):
        if j == -2:
            continue
        elif j == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            n = ekf_state['num_landmarks']
            sigma = ekf_state['P']
            zhat, H = laser_measurement_model(ekf_state, n-1)
            K = np.matmul(np.matmul(sigma, H.T), np.linalg.inv(np.matmul(np.matmul(H, sigma), H.T) + Q))
            z = np.zeros((2, 1))
            z[0,0] = trees[i][0]
            z[1,0] = trees[i][1]
            state_update = np.matmul(K, (z - zhat))
            ekf_state['x'] = np.reshape((np.reshape(ekf_state['x'], (-1, 1)) + state_update), (-1,))
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(ekf_state['x'].shape[0]) - np.matmul(K, H)), sigma))
        else:
            sigma = ekf_state['P']
            zhat,H = laser_measurement_model(ekf_state, j)
            K = np.matmul(np.matmul(sigma, H.T), np.linalg.inv(np.matmul(np.matmul(H, sigma), H.T) + Q))
            z = np.zeros((2, 1))
            z[0,0] = trees[i][0]
            z[1,0] = trees[i][1]
            state_update = np.matmul(K, (z - zhat))
            ekf_state['x'] = np.reshape((np.reshape(ekf_state['x'], (-1,1)) + state_update), (-1,))
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(ekf_state['x'].shape[0]) - np.matmul(K,H)), sigma))

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))
            print("shape: ", ekf_state['x'].shape)

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50,
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0,
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()

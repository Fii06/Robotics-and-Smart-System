# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.1  # Time step
n = 50  # Number of steps
initial_position = 0  # Initial position of the robot
initial_velocity = 1  # Initial velocity of the robot

# Kalman Filter Parameters
A = np.array([[1, dt], [0, 1]])  # State transition model
H = np.array([1, 0]).reshape(1, -1)  # Measurement model, reshaped to (1, 2)
R = 0.1  # Measurement noise covariance
Q = np.array([[1, 0], [0, 1]])  # Process noise covariance
P = np.eye(2)  # Initial state covariance
x = np.array([initial_position, initial_velocity]).reshape(-1, 1)  # Initial state, reshaped to (2, 1)

# Arrays to store results
positions = []
velocities = []
measurements = []

# Simulate Robot Movement with Noise
for i in range(n):
    true_position = initial_position + initial_velocity * i * dt
    measurement = true_position + np.random.normal(0, np.sqrt(R))
    
    # Kalman Prediction and Update
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q
    
    K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
    x = x + K * (measurement - np.dot(H, x))
    P = np.dot(np.eye(2) - np.dot(K, H), P)
    
    positions.append(x[0, 0])  # Estimated position
    velocities.append(x[1, 0])  # Estimated velocity
    measurements.append(measurement)  # Measured position

# Particle Filter Parameters
num_particles = 100
particles = np.random.normal(initial_position, 1, num_particles)
weights = np.ones(num_particles) / num_particles  # Initial weights

particle_positions = []
for i in range(n):
    true_position = initial_position + initial_velocity * i * dt
    measurement = true_position + np.random.normal(0, np.sqrt(R))
    
    particles += np.random.normal(0, 0.1, num_particles)
    
    weights = np.exp(-0.5 * (particles - measurement)**2 / R)
    weights /= np.sum(weights)  # Normalize
    
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]
    particle_positions.append(np.mean(particles))  # Estimated position from particles

# IMU and Lidar Simulation (Simple Model)
imu_angle = np.zeros(n)
lidar_distances = np.zeros(n)

for i in range(1, n):
    imu_angle[i] = imu_angle[i-1] + np.random.normal(0, 0.1)
    lidar_distances[i] = lidar_distances[i-1] + np.random.normal(0, 0.2)

# Extended Kalman Filter (EKF)
def ekf_predict(x, P, u, A, Q):
    x = np.dot(A, x) + u
    P = np.dot(np.dot(A, P), A.T) + Q
    return x, P

control_input = np.array([0.5, 0.5]).reshape(-1, 1)
x_ekf = np.array([initial_position, 0]).reshape(-1, 1)
P_ekf = np.eye(2)

x_ekf, P_ekf = ekf_predict(x_ekf, P_ekf, control_input, A, Q)

# Particle Filter Navigation Implementation
particle_positions_nav = []
for i in range(n):
    true_position = initial_position + initial_velocity * i * dt
    measurement = true_position + np.random.normal(0, np.sqrt(R))
    
    particles += np.random.normal(0, 0.1, num_particles)
    
    weights = np.exp(-0.5 * (particles - measurement)**2 / R)
    weights /= np.sum(weights)  # Normalize
    
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]
    particle_positions_nav.append(np.mean(particles))

# Plotting
plt.figure(figsize=(15, 10))

# Kalman Filter Position Estimation
plt.subplot(231)
plt.plot(positions, label='Estimated Position (Kalman)', color='b')
plt.plot([initial_position + initial_velocity * i * dt for i in range(n)], label='True Position', color='g', linestyle='--')
plt.scatter(range(n), measurements, label='Measured Position', color='r', alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.title('Kalman Filter Position Estimation')
plt.legend()
plt.grid(True)

# Particle Filter Position Estimation
plt.subplot(232)
plt.plot(particle_positions, label='Estimated Position (Particle Filter)', color='b')
plt.plot([initial_position + initial_velocity * i * dt for i in range(n)], label='True Position', color='g', linestyle='--')
plt.scatter(range(n), [initial_position + initial_velocity * i * dt + np.random.normal(0, np.sqrt(R)) for i in range(n)], label='Measured Position', color='r', alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.title('Particle Filter Position Estimation')
plt.legend()
plt.grid(True)

# IMU and Lidar Data
plt.subplot(233)
plt.plot(imu_angle, label='IMU Angle Data', color='c')
plt.plot(lidar_distances, label='LIDAR Distance Data', color='m')
plt.xlabel('Time Steps')
plt.ylabel('Sensor Readings')
plt.title('Localization using IMU and LIDAR')
plt.legend()
plt.grid(True)

# Extended Kalman Filter Navigation
plt.subplot(234)
plt.plot([i * dt for i in range(n)], [x_ekf[0] for _ in range(n)], label='EKF Position Estimate')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.title('Extended Kalman Filter Navigation')
plt.legend()
plt.grid(True)

# Particle Filter Navigation
plt.subplot(235)
plt.plot(particle_positions_nav, label='Particle Filter Navigation', color='b')
plt.plot([initial_position + initial_velocity * i * dt for i in range(n)], label='True Position', color='g', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.title('Particle Filter Navigation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

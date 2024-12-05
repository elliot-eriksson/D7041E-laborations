import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class ESN:
    def root_mean_square_error(self, y, y_hat):
        return np.sqrt(np.mean(np.square(y - y_hat)))
    

    def __init__(self, input_dim, reservoir_size_Nx, output_dim, spectral_radius=0.8, input_scaling=0.2, seed=42):
        self.input_dim = input_dim
        self.reservoir_size_Nx = reservoir_size_Nx
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        np.random.seed(seed)

        self.W_in = (np.random.uniform(-1, 1, [self.reservoir_size_Nx, self.input_dim + 1])) * self.input_scaling
        # Generate W
        self.W = np.random.uniform(-1, 1, [self.reservoir_size_Nx, self.reservoir_size_Nx])
        # Calculate the spectral radius of W (eigenvalues) and divide W by it
        eigenvalues = linalg.eigvals(self.W)
        max_abs_eigenvalue = max(abs(eigenvalues))
        print("Spectral radius of W: ", max_abs_eigenvalue)
        self.W = self.W / max_abs_eigenvalue
        # Scale the W matrix with "ultimate" spectral radius
        self.W = self.W * self.spectral_radius
        self.W_out = None
        
    def _reservoir_update(self, x, u):
        input_data = np.concatenate((np.array([1]), x))
        return np.tanh(np.dot(self.W_in, input_data) + np.dot(self.W, u))
    
         # Concatenate input with bias
        # u_bias = np.append(x, 1)
        # # Update reservoir state
        # new_state = np.tanh(np.dot(self.W_in, u_bias) + np.dot(self.W, u))
        # return new_state
    
    def train_ESN(self, input_data, Y_target, discard_steps, reg_param):
        time_steps = input_data.shape[0]
        X = np.zeros((1+ self.input_dim + self.reservoir_size_Nx, time_steps))

        print("Time steps: ", time_steps)

        reservoir_states = np.zeros((time_steps, self.reservoir_size_Nx))

        x = np.zeros((self.reservoir_size_Nx))

        for t in range(time_steps):
            u = input_data[t]
            x = self._reservoir_update(u, x)
            reservoir_states[t] = x
            

        augmented_reservoir_state = np.hstack([np.ones((time_steps, 1)),input_data, reservoir_states])

        X = augmented_reservoir_state[discard_steps:, :]    
        Y_target = Y_target[discard_steps:, :]
        
        beta = reg_param 
        I = np.eye(X.shape[1])
        self.W_out = np.linalg.solve(np.dot(X, X.T) + beta * I, np.dot(X, Y_target))

    def predict(self, input_data, testing_steps):
        time_steps = input_data.shape[0]
        reservoir_states = np.zeros((time_steps, self.reservoir_size_Nx))
        x = np.zeros((self.reservoir_size_Nx))

        predictions = []

        for t in range(time_steps):
            u = input_data[t]
            x = self._reservoir_update(x, u)
            reservoir_states[t] = x

        # Augment reservoir states with input
        augmented_states = np.hstack([np.ones((time_steps, 1)), input_data, reservoir_states])

        return augmented_states @ self.W_out
        

def sinusoidal_signal(n):
    """Generates a sinusoidal signal.
    n : current time step"""
    
    n = np.arange(1, n + 1)
    return 0.5 * np.sin(n/4)



def test_ESN():
    total_time_steps = 4000
    train_steps = 3000
    test_steps = total_time_steps - train_steps
    reservoir_size = 1000
    spectral_radius = 0.8
    input_scaling = 0.2
    reg_param = 1e-8

    esn = ESN(input_dim=1, 
              reservoir_size_Nx=reservoir_size, 
              output_dim=1, 
              spectral_radius=spectral_radius, 
              input_scaling=input_scaling,
              seed=42)

    signal = sinusoidal_signal(total_time_steps)
    # train_input = signal[:train_steps]
    # test_input = signal[train_steps:]
    # train_target = signal[1:train_steps+1]
    # test_target = signal[train_steps+1:]

    train_input = signal[:3000]
    test_input = signal[3000:]
    train_target = signal[1:3001]  # y(n) depends on y(n-1)
    test_target = signal[3001:]

    esn.train_ESN(train_input, train_target , discard_steps=1000, reg_param=1e-6)
    
    predictions = esn.predict(test_input, test_steps)

    
    plt.plot(test_target, label="True Signal")
    plt.plot(predictions, label="ESN Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
   test_ESN()
    # Plot results

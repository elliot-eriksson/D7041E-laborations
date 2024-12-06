import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from mackey_glass_gen import mackey_glass

class ESN:
    def root_mean_square_error(self, y, y_hat):
        return np.sqrt(np.mean(np.square(y - y_hat)))
    

    def __init__(self, input_dim, reservoir_size_Nx, output_dim, spectral_radius=0.8, input_scaling=0.2, seed=42):
        self.input_dim = input_dim
        self.reservoir_size_Nx = reservoir_size_Nx
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.last_training_state = None
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
        
    def reservoir_update(self, u, x):
        # input_data = np.concatenate((np.array([1]), x))
        # return np.tanh(np.dot(self.W_in, input_data) + np.dot(self.W, u))

        u_bias = np.append(u, 1)
        print(f"u shape: {u.shape}")
        print(f"u_bias shape: {u_bias.shape}")
        print(f"self.W_in shape: {self.W_in.shape}")

        new_state = np.tanh(np.dot(self.W_in, u_bias) + np.dot(self.W, x))
        return new_state
    
    def train_ESN(self, input_data, Y_target, discard_steps, reg_param):
        time_steps = input_data.shape[0]
        X = np.zeros((1+ self.input_dim + self.reservoir_size_Nx, time_steps))

        print("Time steps: ", time_steps)

        reservoir_states = np.zeros((time_steps, self.reservoir_size_Nx))

        x = np.zeros((self.reservoir_size_Nx))

        for t in range(time_steps):
            # u = input_data[t]
            # Sune testar
            if t == 0:
                # For the first time step, no previous output is available
                u = input_data[t]
            else:
                # Use teaching forcing signal
                u = Y_target[t - 1]  # y(n-1)
            #Equation 3 section 2
            #Evalutes the current state and previous reservoir state
            print(f"u shape training: {u.shape}")
            x = self.reservoir_update(u, x)
            reservoir_states[t] = x
            
        self.last_training_state = x
        augmented_reservoir_state = np.hstack([np.ones((time_steps, 1)), input_data.reshape(-1, 1), reservoir_states])

        # augmented_reservoir_state = np.hstack([np.ones((time_steps, 1)),input_data, reservoir_states])

        X = augmented_reservoir_state[discard_steps:, :]    
        Y_target = Y_target[discard_steps:]
        
        beta = reg_param 
        I = np.eye(X.shape[1])
        # self.W_out = np.linalg.solve(np.dot(X, X.T) + beta * I, np.dot(X, Y_target))
        print("passed ")

        # equation 9 in the paper gave us the following
        self.W_out = np.linalg.solve(X.T @ X + beta * I, X.T @ Y_target)


    def predict(self, input_data, testing_steps):
        time_steps = input_data.shape[0]
        # reservoir_states = np.zeros((time_steps, self.reservoir_size_Nx))
        
        reservoir_state = self.last_training_state
        # input data [3000:]
        predictions = []

        current_input = input_data[0] # u
        
        for _ in range(testing_steps):
            # u = input_data[t]
            reservoir_state = self.reservoir_update(current_input, reservoir_state)
            augmented_state = np.hstack([1, current_input, reservoir_state])
            prediction = augmented_state @ self.W_out
            # x = self.reservoir_update(current_input, x)
            # reservoir_states[t] = x
            predictions.append(prediction)
            current_input = prediction
            # x_prev =  x[t-1]

        # Augment reservoir states with input
        # augmented_states = np.hstack([np.ones((time_steps, 1)), input_data.reshape(-1, 1), reservoir_states])

        # augmented_states = np.hstack([np.ones((time_steps, 1)), input_data, reservoir_states])
        return np.array(predictions)
        # return augmented_states @ self.W_out 

    def trainLeaf(self, x_data, y_train_onehot):
        """
        Train the Echo State Network (ESN).
        """

        X_train_with_bias = np.hstack((np.ones((x_data.shape[0], 1)), x_data))
        states = np.zeros((x_data.shape[0], self.nr_reservoir))
        self.states = np.zeros((self.nr_reservoir,))
        for i in range(X_train_with_bias.shape[0]):
            u = X_train_with_bias[i]
            self.states = np.tanh(np.dot(self.Win, u) + np.dot(self.W, self.states))
            states[i] = self.states

        ridge = Ridge(alpha=self.reg_coefficient, fit_intercept=False)
        ridge.fit(states, y_train_onehot)
        self.ridge = ridge
        self.Wout = ridge.coef_

        #######################################################################

        time_steps = input_data.shape[0]
        X = np.zeros((1+ self.input_dim + self.reservoir_size_Nx, time_steps))

        print("Time steps: ", time_steps)

        reservoir_states = np.zeros((time_steps, self.reservoir_size_Nx))

        x = np.zeros((self.reservoir_size_Nx))

        for t in range(time_steps):
            # u = input_data[t]
            # Sune testar
            if t == 0:
                # For the first time step, no previous output is available
                u = input_data[t]
            else:
                # Use teaching forcing signal
                u = Y_target[t - 1]  # y(n-1)
            #Equation 3 section 2
            #Evalutes the current state and previous reservoir state
            print(f"u shape training: {u.shape}")
            x = self.reservoir_update(u, x)
            reservoir_states[t] = x
            
        self.last_training_state = x
        augmented_reservoir_state = np.hstack([np.ones((time_steps, 1)), input_data.reshape(-1, 1), reservoir_states])

        # augmented_reservoir_state = np.hstack([np.ones((time_steps, 1)),input_data, reservoir_states])

        X = augmented_reservoir_state[discard_steps:, :]    
        Y_target = Y_target[discard_steps:]
        
        beta = reg_param 
        I = np.eye(X.shape[1])
        # self.W_out = np.linalg.solve(np.dot(X, X.T) + beta * I, np.dot(X, Y_target))
        print("passed ")

        # equation 9 in the paper gave us the following
        self.W_out = np.linalg.solve(X.T @ X + beta * I, X.T @ Y_target)
    

def sinusoidal_signal(n):
    """Generates a sinusoidal signal.
    n : current time step"""
    
    n = np.arange(1, n + 1)
    return 0.5 * np.sin(n/4)


# 3. Modeling of dynamic systems with ESNs
def Sinusoid_ESN_testing():
    total_time_steps = 4000
    train_steps = 3000
    test_steps = total_time_steps - train_steps
    reservoir_size = 1000
    spectral_radius = 0.8
    input_scaling = 0.2
    reg_param = 1e-8
    nr_of_simulations = 10

    predictions = np.zeros((nr_of_simulations, test_steps))
    # predictions = []

    for run in range(nr_of_simulations):
        esn = ESN(input_dim=1, 
                reservoir_size_Nx=reservoir_size, 
                output_dim=1, 
                spectral_radius=spectral_radius, 
                input_scaling=input_scaling,
                seed=run)

        signal = sinusoidal_signal(total_time_steps)
        train_input = signal[:train_steps]
        test_input = signal[train_steps:]
        train_target = signal[1:train_steps+1]
        test_target = signal[train_steps+1:]

        esn.train_ESN(train_input, train_target , discard_steps=1000, reg_param=reg_param)

        prediction = esn.predict(test_input, test_steps)
        predictions[run] = prediction

        
    mean_predictions = np.mean(predictions, axis=0)
    # print("Root mean square error: ", esn.root_mean_square_error(test_target, mean_predictions))
    print(f'{predictions.shape=}')
    print("Mean predictions: ", mean_predictions)
    
    plt.plot(test_target, label="True Signal")
    plt.plot(mean_predictions, label="ESN Predictions")
    plt.legend()
    plt.show()


# 3. Modeling of dynamic systems with ESNs
def Mackey_Glass_ESN_testing():
    total_time_steps = 4000
    train_steps = 3000
    test_steps = total_time_steps - train_steps
    reservoir_size = 1000
    spectral_radius = 0.8
    input_scaling = 0.2
    reg_param = 1e-8
    nr_of_simulations = 10
    # Todo: ta bort lite grejer här
    signal = mackey_glass(17,total_time_steps)

    predictions = np.zeros((nr_of_simulations, test_steps))
    # predictions = []

    for run in range(nr_of_simulations):
        esn = ESN(input_dim=1, 
                reservoir_size_Nx=reservoir_size, 
                output_dim=1, 
                spectral_radius=spectral_radius, 
                input_scaling=input_scaling,
                seed=run)

        train_input = signal[:train_steps]
        test_input = signal[train_steps:]
        train_target = signal[1:train_steps+1]
        test_target = signal[train_steps+1:]

        esn.train_ESN(train_input, train_target , discard_steps=1000, reg_param=reg_param)

        prediction = esn.predict(test_input, test_steps)
        predictions[run] = prediction

        
    mean_predictions = np.mean(predictions, axis=0)

    print(f'{predictions.shape=}')
    print("Mean predictions: ", mean_predictions)
    
    plt.plot(test_target, label="True Signal")
    plt.plot(mean_predictions, label="ESN Predictions")
    plt.legend()
    plt.show()

    print("Root mean square error: ", esn.root_mean_square_error(test_target, mean_predictions))






##Load the tsv files from the swedishLeaf dataset
def load_data():
    # Load the data
    train_data = np.loadtxt("SwedishLeaf/SwedishLeaf_TRAIN.tsv", delimiter="\t")
    test_data = np.loadtxt("SwedishLeaf/SwedishLeaf_TEST.tsv", delimiter="\t")

    # Extract the features and labels
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int)

    return X_train, y_train, X_test, y_test
# 4. Time-series classification with ESNs
def swedishLeaf_ESN_testing():
    X_train, y_train, X_test, y_test = load_data()
    
    #Scaler the data manualy
    # print(f'{StandardScaler().fit_transform(X_train).shape=}')
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    print(f'{X_train.shape=}')
    print(f'{y_train.shape=}')

    print(f'{X_test.shape=}')
    print(f'{y_test.shape=}')


    # train_steps = 3000
    # test_steps = total_time_steps - train_steps
    reservoir_size = 800
    spectral_radius = 0.99
    input_scaling = 0.25
    reg_param = 1e-8
    nr_of_simulations = 10
    # Todo: ta bort lite grejer här
    

    # signal = mackey_glass(17,total_time_steps)

    # predictions = np.zeros((nr_of_simulations, test_steps))
    predictions = []

    for run in range(nr_of_simulations):
        esn = ESN(input_dim=128, 
                reservoir_size_Nx=reservoir_size, 
                output_dim=128, 
                spectral_radius=spectral_radius, 
                input_scaling=input_scaling,
                seed=run)

        esn.train_ESN(X_train, y_train , discard_steps=1000, reg_param=reg_param)

        prediction = esn.predict(X_test, y_test)
        predictions.append(prediction)

        
    mean_predictions = np.mean(predictions, axis=0)
    # print("Root mean square error: ", esn.root_mean_square_error(test_target, mean_predictions))
    print(f'{predictions.shape=}')
    print("Mean predictions: ", mean_predictions)
    
    plt.plot(test_target, label="True Signal")
    plt.plot(mean_predictions, label="ESN Predictions")
    plt.legend()
    plt.show()
   


if __name__ == "__main__":
    # Sinusoid_ESN_testing()
    # Mackey_Glass_ESN_testing()
    swedishLeaf_ESN_testing()
    # Plot results
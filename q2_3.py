import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(X_filename, y_filename):
    """
    Load dataset from NumPy binary files.
    """
    X = np.load(X_filename)
    y = np.load(y_filename)
    return X, y

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    # Load dataset
    X, y = load_data("X_N_1000_d_5_sig_0_01.npy", "y_N_1000_d_5_sig_0_01.npy")

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Define fractions for training
    fractions = np.arange(0.1, 1.1, 0.1)

    # Initialize lists to store RMSE values
    train_rmse_list = []
    test_rmse_list = []

    # Train models with different fractions of training data
    for fr in fractions:
        # Determine the number of samples to use
        num_samples = int(fr * X_train.shape[0])

        # Select a fraction of training data
        X_train_frac, y_train_frac = X_train[:num_samples], y_train[:num_samples]

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_frac, y_train_frac)

        # Predict on training set
        y_train_pred = model.predict(X_train_frac)
        train_rmse = calculate_rmse(y_train_frac, y_train_pred)
        train_rmse_list.append(train_rmse)

        # Predict on test set
        y_test_pred = model.predict(X_test)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        test_rmse_list.append(test_rmse)

    # Plot RMSE as a function of the number of training samples
    plt.plot(fractions * X_train.shape[0], train_rmse_list, label='Train RMSE')
    plt.plot(fractions * X_train.shape[0], test_rmse_list, label='Test RMSE')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Number of Training Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

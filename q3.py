def lift(x):
    """
    Convert input vector x to a new vector containing all possible coordinate combinations
    of the form xi * xj, for i, j in {1, ..., d}, with i >= j.
    
    Parameters:
    - x: Input vector of length d
    
    Returns:
    - x_lifted: New vector containing all possible coordinate combinations
    """
    d = len(x)
    x_lifted = []
    for i in range(d):
        for j in range(i+1):  # Iterate over indices j <= i
            x_lifted.append(x[i] * x[j])  # Append xi * xj for all j <= i
    return x + x_lifted


# Example usage
x = [1, 2, 3, 4, 5]
lifted_x = lift(x)
print(lifted_x)

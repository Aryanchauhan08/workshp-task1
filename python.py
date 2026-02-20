import numpy as np

def lu_factorization(A):
    """
    Computes the LU factorization of a square matrix A using Gaussian elimination with partial pivoting.
    
    Parameters:
    A (numpy.ndarray): The input square matrix (n x n).
    
    Returns:
    P (numpy.ndarray): The permutation matrix.
    L (numpy.ndarray): The lower triangular matrix with unit diagonal.
    U (numpy.ndarray): The upper triangular matrix.
    """
    n = A.shape[0]
    
    # Create copies to avoid modifying the original matrix
    U = A.copy().astype(float)
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    
    for k in range(n):
        # 1. Partial Pivoting: Find the index of the largest absolute value in the current column
        pivot_row = np.argmax(np.abs(U[k:, k])) + k
        
        # Check for singularity (if pivot is effectively zero)
        if np.abs(U[pivot_row, k]) < 1e-12:
            raise ValueError("Matrix is singular: No nonzero pivots exist.")
            
        # 2. Row Swapping: Swap rows in U, P, and the computed part of L if needed
        if pivot_row != k:
            U[[k, pivot_row], k:] = U[[pivot_row, k], k:]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                
        # 3. Elimination: Update rows below the pivot using vectorized subtraction
        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            # Update the row: Row_i = Row_i - multiplier * Row_k
            U[i, k:] = U[i, k:] - (multiplier * U[k, k:])
            
    return P, L, U

def forward_substitution(L, b, P):
    """
    Solves the lower triangular system Ly = Pb for y using forward substitution.
    
    Parameters:
    L (numpy.ndarray): The lower triangular matrix.
    b (numpy.ndarray): The right-hand side vector.
    P (numpy.ndarray): The permutation matrix.
    
    Returns:
    y (numpy.ndarray): The solution vector for the intermediate system.
    """
    n = L.shape[0]
    
    # Apply permutation to b: Pb = P * b
    Pb = np.zeros(n)
    for i in range(n):
        Pb[i] = np.sum(P[i, :] * b)
    
    y = np.zeros(n)
    for i in range(n):
        # Calculate y[i] = Pb[i] - sum(L[i,j] * y[j]) using a vectorized dot product
        sum_val = np.sum(L[i, :i] * y[:i])
        y[i] = Pb[i] - sum_val
        
    return y

def backward_substitution(U, y):
    """
    Solves the upper triangular system Ux = y for x using backward substitution.
    
    Parameters:
    U (numpy.ndarray): The upper triangular matrix.
    y (numpy.ndarray): The intermediate solution vector.
    
    Returns:
    x (numpy.ndarray): The final solution vector.
    """
    n = U.shape[0]
    x = np.zeros(n)
    
    # Iterate backwards from the last row down to the first
    for i in range(n - 1, -1, -1):
        # Calculate sum(U[i,j] * x[j]) for known x values using a vectorized dot product
        sum_val = np.sum(U[i, i+1:] * x[i+1:])
        x[i] = (y[i] - sum_val) / U[i, i]
        
    return x

def calculate_backward_error(P, A, L, U):
    """
    Computes the relative backward error of the factorization: ||PA - LU|| / ||A||.
    
    Parameters:
    P, A, L, U (numpy.ndarray): The matrices from the factorization PA = LU.
    
    Returns:
    float: The Frobenius norm of the error matrix normalized by the norm of A.
    """
    norm_A = np.linalg.norm(A, ord='fro')
    if norm_A == 0: return 0.0
    
    # Calculate the product matrices PA and LU
    PA = np.matmul(P, A)
    LU = np.matmul(L, U)
    
    # Compute the difference and its norm
    error_matrix = PA - LU
    return np.linalg.norm(error_matrix, ord='fro') / norm_A

def calculate_residual(A, x_hat, b):
    """
    Computes the relative residual of the solution: ||Ax - b|| / ||b||.
    
    Parameters:
    A (numpy.ndarray): The original coefficient matrix.
    x_hat (numpy.ndarray): The computed solution vector.
    b (numpy.ndarray): The original right-hand side vector.
    
    Returns:
    float: The L2 norm of the residual vector normalized by the norm of b.
    """
    norm_b = np.linalg.norm(b, ord=2)
    if norm_b == 0: return 0.0
    
    # Calculate A * x_hat using row-wise dot products
    Ax = np.zeros_like(b)
    for i in range(len(b)):
        Ax[i] = np.sum(A[i, :] * x_hat)
        
    # Compute the residual vector (Ax - b) and its norm
    residual_vector = Ax - b
    return np.linalg.norm(residual_vector, ord=2) / norm_b

print("hi")

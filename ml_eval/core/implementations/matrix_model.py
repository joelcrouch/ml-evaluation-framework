
from typing import List, Dict, Any

def multiply_matrices(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """
    Multiplies two matrices.
    """
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Number of columns in Matrix A must be equal to number of rows in Matrix B")

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result

class MatrixMultiplicationModel:
    """
    A simple simulator for a matrix multiplication model.
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates running a matrix multiplication model.
        Expects 'matrix_a' and 'matrix_b' in input_data.
        """
        matrix_a = input_data.get("matrix_a")
        matrix_b = input_data.get("matrix_b")

        if not matrix_a or not matrix_b:
            raise ValueError("Input data must contain 'matrix_a' and 'matrix_b'.")
        
        if not isinstance(matrix_a, list) or not all(isinstance(row, list) for row in matrix_a):
            raise ValueError("'matrix_a' must be a list of lists.")
        if not isinstance(matrix_b, list) or not all(isinstance(row, list) for row in matrix_b):
            raise ValueError("'matrix_b' must be a list of lists.")

        result_matrix = multiply_matrices(matrix_a, matrix_b)
        return {"result_matrix": result_matrix}


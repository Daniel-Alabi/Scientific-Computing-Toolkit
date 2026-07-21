"""Dense linear algebra on plain Python lists.

Matrices are lists of row-lists; vectors are flat lists of floats.
"""

from __future__ import annotations

from typing import List, Tuple

Matrix = List[List[float]]
Vector = List[float]


def _copy(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def gauss_solve(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]  # augmented
    for k in range(n):
        # partial pivot
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-15:
            raise ValueError("matrix is singular or nearly singular")
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= m * M[k][j]
    # back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def lu_decompose(A: Matrix) -> Tuple[Matrix, Matrix, List[int]]:
    """LU decomposition with partial pivoting: PA = LU.

    Returns (L, U, perm) where perm[i] is the row of A in position i.
    """
    n = len(A)
    U = _copy(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    perm = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[p][k]) < 1e-15:
            raise ValueError("matrix is singular or nearly singular")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            for j in range(k):
                L[k][j], L[p][j] = L[p][j], L[k][j]
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]
    return L, U, perm


def lu_solve(L: Matrix, U: Matrix, perm: List[int], b: Vector) -> Vector:
    """Solve Ax = b given the LU factors of A."""
    n = len(L)
    pb = [b[perm[i]] for i in range(n)]
    # forward: Ly = Pb
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))
    # backward: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def jacobi(
    A: Matrix, b: Vector, x0: Vector | None = None, tol: float = 1e-10, max_iter: int = 10_000
) -> Vector:
    """Jacobi iteration. Converges for strictly diagonally dominant A."""
    n = len(A)
    x = list(x0) if x0 is not None else [0.0] * n
    for _ in range(max_iter):
        x_new = [
            (b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / A[i][i]
            for i in range(n)
        ]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new
        x = x_new
    raise RuntimeError(f"jacobi did not converge in {max_iter} iterations")


def gauss_seidel(
    A: Matrix, b: Vector, x0: Vector | None = None, tol: float = 1e-10, max_iter: int = 10_000
) -> Vector:
    """Gauss-Seidel iteration; typically ~2x faster than Jacobi when it converges."""
    n = len(A)
    x = list(x0) if x0 is not None else [0.0] * n
    for _ in range(max_iter):
        delta = 0.0
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            new = (b[i] - s) / A[i][i]
            delta = max(delta, abs(new - x[i]))
            x[i] = new
        if delta < tol:
            return x
    raise RuntimeError(f"gauss_seidel did not converge in {max_iter} iterations")


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix product A @ B."""
    n, m, p = len(A), len(B[0]), len(B)
    if len(A[0]) != p:
        raise ValueError("incompatible shapes")
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]


def matvec(A: Matrix, x: Vector) -> Vector:
    """Matrix-vector product A @ x."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def transpose(A: Matrix) -> Matrix:
    return [list(col) for col in zip(*A)]


def determinant(A: Matrix) -> float:
    """Determinant via LU decomposition, O(n^3)."""
    n = len(A)
    try:
        _, U, perm = lu_decompose(A)
    except ValueError:
        return 0.0
    det = 1.0
    for i in range(n):
        det *= U[i][i]
    # sign of permutation
    visited = [False] * n
    sign = 1
    for i in range(n):
        if not visited[i]:
            j, cycle = i, 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle += 1
            if cycle % 2 == 0:
                sign = -sign
    return sign * det


def norm(x: Vector, p: float = 2) -> float:
    """Vector p-norm (p = 1, 2, or float('inf'))."""
    if p == float("inf"):
        return max(abs(v) for v in x)
    return sum(abs(v) ** p for v in x) ** (1.0 / p)

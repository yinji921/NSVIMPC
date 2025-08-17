import numpy as np


def nnls(A, b, eps: float = 1.0e-10, max_steps: int = 100):
    # non-negative least-squares after
    # <https://en.wikipedia.org/wiki/Non-negative_least_squares>
    A = np.asarray(A)
    b = np.asarray(b)

    AtA = A.T @ A
    Atb = A.T @ b

    m, n = A.shape
    assert m == b.shape[0]
    mask = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    w = Atb
    s = np.zeros(n)
    k = 0
    while sum(mask) != n and max(w) > eps:
        if k >= max_steps:
            break
        mask[np.argmax(w)] = True

        s[mask] = np.linalg.lstsq(AtA[mask][:, mask], Atb[mask], rcond=None)[0]
        s[~mask] = 0.0

        while np.min(s[mask]) <= 0:
            alpha = np.min(x[mask] / (x[mask] - s[mask]))
            x += alpha * (s - x)
            mask[np.abs(x) < eps] = False

            s[mask] = np.linalg.lstsq(AtA[mask][:, mask], Atb[mask], rcond=None)[0]
            s[~mask] = 0.0

        x = s.copy()
        w = Atb - AtA @ x

        k += 1

    return x



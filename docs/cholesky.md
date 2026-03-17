
`cholesky_solve()`
to solve:

[
LL^T x = b
]

1. forward substitution：`L y = b`
2. backward substitution：`L^T x = y`


## forward substitution

[
y_i = \frac{b_i - \sum_{k=0}^{i-1} L_{ik} y_k}{L_{ii}}
]

## backward substitution

[
x_i = \frac{y_i - \sum_{k=i+1}^{n-1} L_{ki} x_k}{L_{ii}}
]

because `L^T[i,k] = L[k,i]`。

---

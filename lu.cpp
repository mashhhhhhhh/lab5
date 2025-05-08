#include <vector>

void lu_decompose(const std::vector<std::vector<double>>& mat,
    std::vector<std::vector<double>>& L,
    std::vector<std::vector<double>>& U) {
    int n = mat.size();
    L.assign(n, std::vector<double>(n, 0.0));
    U.assign(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int k = i; k < n; ++k) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) sum += L[i][j] * U[j][k];
            U[i][k] = mat[i][k] - sum;
        }

        for (int k = i; k < n; ++k) {
            if (i == k) L[i][i] = 1.0;
            else {
                double sum = 0.0;
                for (int j = 0; j < i; ++j) sum += L[k][j] * U[j][i];
                L[k][i] = (mat[k][i] - sum) / U[i][i];
            }
        }
    }
}

std::vector<double> solve_lu(const std::vector<std::vector<double>>& mat,
    const std::vector<double>& rhs) {
    int n = mat.size();
    std::vector<std::vector<double>> L, U;
    lu_decompose(mat, L, U);

    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) sum += L[i][j] * y[j];
        y[i] = rhs[i] - sum;
    }

    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }

    return x;
}

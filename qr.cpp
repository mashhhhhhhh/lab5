#include <vector>
#include <cmath>

void qr_decompose(const std::vector<std::vector<double>>& mat,
    std::vector<std::vector<double>>& Q,
    std::vector<std::vector<double>>& R) {
    int n = mat.size();
    Q.assign(n, std::vector<double>(n, 0.0));
    R.assign(n, std::vector<double>(n, 0.0));

    for (int j = 0; j < n; ++j) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = mat[i][j];

        for (int k = 0; k < j; ++k) {
            R[k][j] = 0.0;
            for (int i = 0; i < n; ++i) R[k][j] += Q[i][k] * mat[i][j];
            for (int i = 0; i < n; ++i) v[i] -= R[k][j] * Q[i][k];
        }

        R[j][j] = 0.0;
        for (int i = 0; i < n; ++i) R[j][j] += v[i] * v[i];
        R[j][j] = sqrt(R[j][j]);

        for (int i = 0; i < n; ++i) Q[i][j] = v[i] / R[j][j];
    }
}

std::vector<double> solve_qr(const std::vector<std::vector<double>>& mat,
    const std::vector<double>& rhs) {
    int n = mat.size();
    std::vector<std::vector<double>> Q, R;
    qr_decompose(mat, Q, R);

    std::vector<double> qt_f(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            qt_f[i] += Q[j][i] * rhs[j];
        }
    }

    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += R[i][j] * x[j];
        x[i] = (qt_f[i] - sum) / R[i][i];
    }

    return x;
}
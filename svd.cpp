#include <vector>
#include <cmath>
#include <algorithm>

void jacobi_svd(const std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& U,
    std::vector<double>& S,
    std::vector<std::vector<double>>& V) {
    const int n = A.size();
    const double eps = 1e-12;
    const int max_iter = 100;

    U = A;
    S.assign(n, 0.0);
    V.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_off = 0.0;
        int p = 0, q = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double off = fabs(U[i][j]) + fabs(U[j][i]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < eps) break;

        double theta = 0.5 * atan2(2 * U[p][q], U[q][q] - U[p][p]);
        double c = cos(theta), s = sin(theta);

        for (int j = 0; j < n; ++j) {
            double upj = U[p][j], uqj = U[q][j];
            U[p][j] = c * upj - s * uqj;
            U[q][j] = s * upj + c * uqj;

            double vjp = V[j][p], vjq = V[j][q];
            V[j][p] = c * vjp - s * vjq;
            V[j][q] = s * vjp + c * vjq;
        }
    }

    for (int i = 0; i < n; ++i) S[i] = fabs(U[i][i]);
    std::sort(S.begin(), S.end(), std::greater<double>());
}

std::vector<double> compute_svd_values(const std::vector<std::vector<double>>& A) {
    std::vector<std::vector<double>> U, V;
    std::vector<double> S;
    jacobi_svd(A, U, S, V);
    return S;
}

double compute_cond_number(const std::vector<std::vector<double>>& A) {
    auto sv = compute_svd_values(A);
    if (sv.empty() || sv.back() < 1e-12) return 1.0;
    return sv.front() / sv.back();
}
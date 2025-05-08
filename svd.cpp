#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

void jacobi_svd(const vector<vector<double>>& A,
    vector<vector<double>>& U,
    vector<double>& S,
    vector<vector<double>>& V);

vector<double> solve_svd(const vector<vector<double>>& A,
    const vector<double>& f) {
    int n = A.size();
    vector<vector<double>> U, V;
    vector<double> S;

    // Вычисляем SVD
    jacobi_svd(A, U, S, V);

    // Вычисляем U^T * f
    vector<double> ut_f(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ut_f[i] += U[j][i] * f[j];
        }
    }

    // Делим на сингулярные числа
    vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        y[i] = ut_f[i] / S[i];
    }

    // Умножаем на V
    vector<double> x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            x[i] += V[i][j] * y[j];
        }
    }

    return x;
}

vector<double> compute_svd_values(const vector<vector<double>>& A) {
    vector<vector<double>> U, V;
    vector<double> S;
    jacobi_svd(A, U, S, V);
    return S;
}

double compute_cond_number(const vector<vector<double>>& A) {
    auto sv = compute_svd_values(A);
    double cond = sv.front() / sv.back();
    return cond;
}

void jacobi_svd(const vector<vector<double>>& A,
    vector<vector<double>>& U,
    vector<double>& S,
    vector<vector<double>>& V) {
    const int n = A.size();
    const double eps = 1e-12;
    const int max_iter = 100;

    U = A;
    S.assign(n, 0.0);
    V.assign(n, vector<double>(n, 0.0));
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
    sort(S.begin(), S.end(), greater<double>());
}

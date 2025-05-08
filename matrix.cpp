#include <vector>
#include <cmath>

std::vector<std::vector<double>> gen_matrix(int n) {
    std::vector<std::vector<double>> mat(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = 1.0 / (1.0 + 3 * (i + 1) + 2 * (j + 1));
        }
    }
    return mat;
}

std::vector<double> gen_exact_solution(int n) {
    std::vector<double> sol(n, 1.0);
    sol.back() = n;
    return sol;
}

std::vector<double> mat_vec_mul(const std::vector<std::vector<double>>& mat,
    const std::vector<double>& vec) {
    int n = mat.size();
    std::vector<double> res(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

double vector_norm(const std::vector<double>& v) {
    double norm = 0.0;
    for (double x : v) norm += x * x;
    return sqrt(norm);
}

double compute_error(const std::vector<double>& approx,
    const std::vector<double>& exact) {
    int n = approx.size();
    std::vector<double> diff(n);
    for (int i = 0; i < n; ++i) diff[i] = approx[i] - exact[i];
    return vector_norm(diff) / vector_norm(exact);
}
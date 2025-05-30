#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace chrono;
vector<double> solve_svd(const vector<vector<double>>& mat, const vector<double>& rhs);
vector<double> compute_svd_values(const vector<vector<double>>& mat);
double compute_cond_number(const vector<vector<double>>& mat);
vector<vector<double>> gen_matrix(int size);
vector<double> gen_exact_solution(int size);
vector<double> mat_vec_mul(const vector<vector<double>>& mat, const vector<double>& vec);
double compute_error(const vector<double>& approx, const vector<double>& exact);
vector<double> solve_lu(const vector<vector<double>>& mat, const vector<double>& rhs);
vector<double> solve_qr(const vector<vector<double>>& mat, const vector<double>& rhs);
vector<double> compute_svd_values(const vector<vector<double>>& mat);
double compute_cond_number(const vector<vector<double>>& mat);

void print_results(int n, const string& method, double error,
    long long time, const vector<double>& sv, double cond) {
    cout << " " << n << "  " << method << "  "<< error << "  " << time << "  ";

    cout << "[";
    int show_sv = min(3, (int)sv.size());
    for (int i = 0; i < show_sv; ++i) {
        cout << scientific << setprecision(2) << sv[i];
        if (i < show_sv - 1) cout << ",";
    }
    cout << "] "  << cond << " " << endl;
}

void run_test(int n) {
    auto mat = gen_matrix(n);
    auto exact = gen_exact_solution(n);
    auto rhs = mat_vec_mul(mat, exact);

    auto sv_start = high_resolution_clock::now();
    auto sv = compute_svd_values(mat);
    auto cond = compute_cond_number(mat);
    auto svd_time = duration_cast<microseconds>(high_resolution_clock::now() - sv_start).count();

    auto lu_start = high_resolution_clock::now();
    auto x_lu = solve_lu(mat, rhs);
    auto lu_time = duration_cast<microseconds>(high_resolution_clock::now() - lu_start).count();
    double lu_err = compute_error(x_lu, exact);

    auto qr_start = high_resolution_clock::now();
    auto x_qr = solve_qr(mat, rhs);
    auto qr_time = duration_cast<microseconds>(high_resolution_clock::now() - qr_start).count();
    double qr_err = compute_error(x_qr, exact);

    auto svd_start = high_resolution_clock::now();
    auto x_svd = solve_svd(mat, rhs);
    auto full_svd_time = duration_cast<microseconds>(high_resolution_clock::now() - svd_start).count();
    double svd_err = compute_error(x_svd, exact);

    print_results(n, "LU", lu_err, lu_time, sv, cond);
    print_results(n, "QR", qr_err, qr_time, sv, cond);
    print_results(n, "SVD", svd_err, full_svd_time, sv, cond);
}

int main() {
    cout << " Size  Method  Error       Time (μs) Singular Values  Cond(A)    │" << endl;

    for (int n : {5, 10, 20}) {
        run_test(n);
    }

    return 0;
}

#ifndef svm_hpp
#define svm_hpp

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
using namespace std;

namespace kernel {

double polynomial(const vector<double>& x, const vector<double>& y, const vector<double>& hyper_params = {}) {
    if (x.size() != y.size()) {
        cerr << "[ERROR]: number of elements can't be matched for inner product.";
        exit(-500);
    } else if (hyper_params.size() != 2) {
        cerr << "[ERROR]: number of hyper-parameters must be equaled to 2.";
        exit(-501);
    }
    
    double res = 0;
    for (size_t i = 0; i < x.size(); i++) res += x[i] * y[i];
    res += hyper_params[0];
    res = pow(res, hyper_params[1]);
    return res;
}

double histogram_intersection(const vector<double>& x, const vector<double>& y, const vector<double>& hyper_params = {}) {
    if (x.size() != y.size()) {
        cerr << "[ERROR]: number of elements can't be matched for inner product.";
        exit(-500);
    }
    
    double res = 0;
    for (size_t i = 0; i < x.size(); i++) {
        res += min(x[i], y[i]);
    }
    return res;
}

typedef function<double(const vector<double>&, const vector<double>&, const vector<double>&)> kernel_func;
} // kernel namespace


class KernelSVM {
public:
    bool verbose;
    double b;
    vector<vector<double>> xs, xs_in;
    vector<int> ys, ys_in;
    vector<double> alpha_s, alpha_s_in;
    kernel::kernel_func K;
    vector<double> hyper_params;
    
    void log(const string str) {
        if (this->verbose) cout << str << endl;
    }

public:
    double accuracy, accuracy_c1, accuracy_c2;
    size_t correct_c1, correct_c2;
    
    KernelSVM(const kernel::kernel_func _K = kernel::polynomial, const vector<double> _hyper_params={1,0}, const bool _verbose = true) : K(_K), hyper_params(_hyper_params), verbose(_verbose) {}
    KernelSVM() = delete;
    
    void train(const vector<vector<double>>& dataclass1, const vector<vector<double>>& dataclass2, const size_t D, const double C, const double lr, const double limit=10e-4) {
        const double EPS = 10e-7;
        
        double item1, item2, item3, delta, beta, broken;
        vector<vector<double>> x; vector<int> y; vector<double> alpha;
        for (size_t i = 0; i < dataclass1.size(); i++) {
            x.push_back(dataclass1[i]);
            y.push_back(1);
        }
        for (size_t i = 0; i < dataclass2.size(); i++) {
            x.push_back(dataclass2[i]);
            y.push_back(-1);
        }
        size_t N = x.size(), Ns, Ns_in;
        alpha = vector<double>(N, 0);
        beta = 1;
        
        bool judge;
        do {
            judge = false;
            broken = 0.0;
            
            for (size_t i = 0; i < N; i++) {
                item1 = 0; item2 = 0;
                for (size_t j = 0; j < N; j++) {
                    item1 += alpha[j] * double(y[i]) * double(y[j]) * K(x[i], x[j], hyper_params);
                    item2 += alpha[j] * double(y[i]) * double(y[j]);
                }
                
                delta = 1 - item1 - beta * item2;
                
                alpha[i] += lr * delta;
                if (alpha[i] < 0) alpha[i] = 0;
                else if (alpha[i] > C) alpha[i] = C;
                else if (abs(delta) > limit) {
                    judge = true;
                    broken += abs(delta) - limit;
                }
            }
            
            item3 = 0; for (size_t i = 0; i < N; i++) item3 += alpha[i] * double(y[i]);
            beta += item3 * item3 / 2.0;
        } while(judge);
        
        Ns = 0; Ns_in = 0;
        for (size_t i = 0; i < N; i++) {
            if ( (EPS < alpha[i]) && (alpha[i] < C - EPS) ) {
                xs.push_back(x[i]);
                ys.push_back(y[i]);
                alpha_s.push_back(alpha[i]);
                Ns += 1;
            } else if (alpha[i] >= C - EPS) {
                xs_in.push_back(x[i]);
                ys_in.push_back(y[i]);
                alpha_s_in.push_back(alpha[i]);
                Ns_in += 1;
            }
        }
        
        b = 0;
        for (size_t i = 0; i < Ns; i++) {
            b += double(ys[i]);
            for (size_t j = 0; j < Ns; j++)
                b -= alpha_s[j] * double(ys[j]) * K(xs[j], xs[i], hyper_params);
            for (size_t j = 0; j < Ns_in; j++)
                b -= alpha_s_in[j] * double(ys_in[j]) * K(xs_in[j], xs[i], hyper_params);
        }
        b /= double(Ns);
    }
    double h(const vector<double>& x) {
        size_t i; double res = 0;
        for (i = 0; i < xs.size(); i++) res += alpha_s[i] * ys[i] * K(xs[i], x, hyper_params);
        for (i = 0; i < xs_in.size(); i++) res += alpha_s_in[i] * ys_in[i] * K(xs_in[i], x, hyper_params);
        res += b;
        return res;
    }
    double g(const vector<double> x) {
        double hx = h(x);
        if (hx >= 0) return 1;
        else return -1;
    }
    
    void test(const vector<vector<double>>& dataclass1, const vector<vector<double>>& dataclass2) {
        size_t i;
        
        correct_c1 = 0;
        for (i = 0; i < dataclass1.size(); i++)
            if (g(dataclass1[i]) == 1) correct_c1 += 1;
        correct_c2 = 0;
        for (i = 0; i < dataclass2.size(); i++)
            if (g(dataclass2[i]) == -1) correct_c2 += 1;
        
        accuracy = (double)(correct_c1 + correct_c2) / (double)(dataclass1.size() + dataclass2.size());
        accuracy_c1 = (double)(correct_c1) / (double)dataclass1.size();
        accuracy_c2 = (double)(correct_c2) / (double)dataclass2.size();
    }
    
    double predict_for_one(const vector<double>& vec) {
        return g(vec);
    }
};

#endif /* svm_hpp */

#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <queue>
#include "svm.hpp"
#define inout ios::sync_with_stdio(0); cin.tie(nullptr); cout.tie(nullptr)
#define ll long long
#define ull unsigned long long
using namespace std;
using namespace kernel;

int main(int argc, char *argv[]) {
    inout;

    if (argc < 2) {
        cerr << "[ERROR]. Usage: ./svm_entry <image_path> -<train or test>" << endl;
        return -1;
    }
    int mode = 1; // 1 - train, 2 - test, 3 - mode
    string arg = (argv[1]);
    if (arg == "-test") mode = 2;
    else if (arg == "-train") mode = 1;
    else if (arg == "-predict") mode = 3;
    else {
        cerr << "[ERROR]: Wrong parameter: " << arg << endl;
        return -1;
    }
    if (mode == 1) {
        ifstream data0("dataclass0.log");
        size_t size, len; data0 >> size >> len;
        vector<vector<double>> dataclass0(size, vector<double>(len));
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < len; j++) {
                data0 >> dataclass0[i][j];
            }
        }
        ifstream data1("dataclass1.log");
        data1 >> size >> len;
        vector<vector<double>> dataclass1(size, vector<double>(len));
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < len; j++) {
                data1 >> dataclass1[i][j];
            }
        }
        data0.close(); data1.close();
        
        size_t i,j;
        kernel_func K = histogram_intersection;
        KernelSVM svm(K);
        svm.train(dataclass0, dataclass1, 2, 1, 0.0001);
        ofstream temp("__svmcppcache.tmp");
        
        temp << svm.xs.size() << ' ' << svm.xs[0].size() << endl;
        for (i = 0; i < svm.xs.size(); i++) {
            for (j = 0; j < svm.xs[0].size() - 1; j++)
                temp << svm.xs[i][j] << ' ' << endl;
            temp << svm.xs[i].back() << endl;
        }
        temp << svm.xs_in.size() << ' ' << svm.xs_in[0].size() << endl;
        for (i = 0; i < svm.xs_in.size(); i++) {
            for (j = 0; j < svm.xs_in[0].size() - 1; j++)
                temp << svm.xs_in[i][j] << ' ' << endl;
            temp << svm.xs_in[i].back() << endl;
        }
        
        temp << svm.alpha_s.size() << endl;
        for (i = 0; i < svm.alpha_s.size() - 1; i++) {
            temp << svm.alpha_s[i] << ' ' << endl;
        } temp << svm.alpha_s.back() << endl;
        temp << svm.alpha_s_in.size() << endl;
        for (i = 0; i < svm.alpha_s_in.size() - 1; i++) {
            temp << svm.alpha_s_in[i] << ' ' << endl;
        } temp << svm.alpha_s_in.back() << endl;
        
        temp << svm.ys.size() << endl;
        for (i = 0; i < svm.ys.size() - 1; i++) {
            temp << svm.ys[i] << ' ' << endl;
        } temp << svm.ys.back() << endl;
        temp << svm.ys_in.size() << endl;
        for (i = 0; i < svm.ys_in.size() - 1; i++) {
            temp << svm.ys_in[i] << ' ' << endl;
        } temp << svm.ys_in.back() << endl;
        
        temp << svm.b << endl;
        
        temp.close();
    } else if (mode == 2) {
        ifstream temp("__svmcppcache.tmp");
        
        size_t i,j;
        kernel_func K = histogram_intersection;
        KernelSVM svm(K);
        
        size_t size, len;
        
        temp >> size >> len;
        svm.xs = vector<vector<double>>(size, vector<double>(len));
        for (i = 0; i < size; i++) {
            for (j = 0; j < len; j++)
                temp >> svm.xs[i][j];
        }
        temp >> size >> len;
        svm.xs_in = vector<vector<double>>(size, vector<double>(len));
        for (i = 0; i < size; i++) {
            for (j = 0; j < len; j++)
                temp >> svm.xs_in[i][j];
        }
        
        temp >> size;
        svm.alpha_s = vector<double>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.alpha_s[i];
        }
        temp >> size;
        svm.alpha_s_in = vector<double>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.alpha_s_in[i];
        }
        
        temp >> size;
        svm.ys = vector<int>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.ys[i];
        }
        temp >> size;
        svm.ys_in = vector<int>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.ys_in[i];
        }
        
        temp >> svm.b;
        
        temp.close();
        
        ifstream data0("dataclass0.log");
        data0 >> size >> len;
        vector<vector<double>> dataclass0(size, vector<double>(len));
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < len; j++) {
                data0 >> dataclass0[i][j];
            }
        }
        ifstream data1("dataclass1.log");
        data1 >> size >> len;
        vector<vector<double>> dataclass1(size, vector<double>(len));
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < len; j++) {
                data1 >> dataclass1[i][j];
            }
        }
        data0.close(); data1.close();
        
        svm.test(dataclass0, dataclass1);
        
        ofstream accuracy_data("accuracy.log");
        accuracy_data << svm.accuracy << endl << svm.accuracy_c1 << endl << svm.accuracy_c2 << endl;
    } else {
        ifstream temp("__svmcppcache.tmp");
        
        size_t i,j;
        kernel_func K = histogram_intersection;
        KernelSVM svm(K);
        
        size_t size, len;
        
        temp >> size >> len;
        svm.xs = vector<vector<double>>(size, vector<double>(len));
        for (i = 0; i < size; i++) {
            for (j = 0; j < len; j++)
                temp >> svm.xs[i][j];
        }
        temp >> size >> len;
        svm.xs_in = vector<vector<double>>(size, vector<double>(len));
        for (i = 0; i < size; i++) {
            for (j = 0; j < len; j++)
                temp >> svm.xs_in[i][j];
        }
        
        temp >> size;
        svm.alpha_s = vector<double>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.alpha_s[i];
        }
        temp >> size;
        svm.alpha_s_in = vector<double>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.alpha_s_in[i];
        }
        
        temp >> size;
        svm.ys = vector<int>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.ys[i];
        }
        temp >> size;
        svm.ys_in = vector<int>(size);
        for (i = 0; i < size; i++) {
            temp >> svm.ys_in[i];
        }
        
        temp >> svm.b;
        
        temp.close();
        
        ifstream data("feature.log");
        data >> size;
        vector<double> feature(size);
        for (size_t i = 0; i < size; i++) {
            data >> feature[i];
        }
        data.close();
        
        int result = svm.predict_for_one(feature);
        
        ofstream predict("predict.log");
        predict << result << endl;
    }
}

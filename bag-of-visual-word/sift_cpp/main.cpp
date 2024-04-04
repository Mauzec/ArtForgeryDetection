#define _SILENCE_CXX20_CISO646_REMOVED_WARNING
#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <queue>
#include "sift.hpp"
#define ll long long
#define ull unsigned long long
using namespace std;

int main(int argc, char *argv[]) {
//    std::ofstream f("example.json");
//    
//    json ex1 = json::parse(R"(
//      {
//        "pi": 3.141,
//        "happy": true
//      }
//    )");
//    
//    json ex2 = {
//      {"pi", 3.141},
//      {"happy", true},
//      {"name", "Niels"},
//      {"nothing", nullptr},
//      {"answer", {
//        {"everything", 42}
//      }},
//      {"list", {1, 0, 2}},
//      {"object", {
//        {"currency", "USD"},
//        {"value", 42.99}
//      }}
//    };
//    
//    f << setw(4) << ex2 << endl;
//    
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    if (argc < 2) {
        std::cerr << "[ERROR]. Usage: ./sift <image_path> -drawkps=<0 or 1>. Last parameter -drawkps is optional";
        return 0;
    }
    bool return_kps_image = false;
    bool wrong = false;
    if (argc >= 3) {
        size_t size = strlen(argv[2]);
        if (size != 10 || (argv[2][9] != '0' && argv[2][9] != '1')) {
            cout << "[ERROR]: Given wrong parameter " << argv[2] << endl;
            wrong = true;
        }
        if (!wrong) {
            const char* drawkps = "-drawkps=";
            for (int i = 1; i < 9; i++) {
                if (drawkps[i] != argv[2][i]) {
                    cout << "[ERROR]: Given wrong parameter " << argv[2] << endl;
                    wrong = true;
                    break;
                }
            }
        }
        if (!wrong) {
//            cout << argv[2][9] << endl;
            if (argv[2][9] == '0') return_kps_image = false;
            else return_kps_image = true;
        }
    }
    if (wrong) return 0;
    
    bool add_to_name = false;
    if (argc == 4) add_to_name = true;
    
    Image img(argv[1]);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);
    std::vector<sift::Keypoint> kps = add_to_name ? sift::find_keypoints_and_descriptors(img, true, "kps" + string(argv[3]) + ".json") : sift::find_keypoints_and_descriptors(img);
    if (return_kps_image) {
        Image result = sift::draw_keypoints(img, kps);
        string result_path = "result.png";
        if (add_to_name)
            result_path = "result" + string(argv[3]) + ".jpg";
        result.save(result_path);
        std::cout << "Output image is saved as " << result_path  << endl;
    }
//    for (auto& des : kps) cout << des.x << ' ' << des.y << endl;
    
    std::cout << "Found " << kps.size() << " keypoints" << endl;
    
    
    return 0;
}

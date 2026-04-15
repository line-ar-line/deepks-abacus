#include "hydrogen.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

// 读取输入文件参数
void readInputFile(const string& filename, int& num_levels, bool& generate_density, 
                   double& r_max, int& r_points) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "无法打开输入文件 " << filename << endl;
        // 使用默认值
        num_levels = 5;
        generate_density = false;
        r_max = 20.0;
        r_points = 1000;
        return;
    }
    
    string line;
    while (getline(infile, line)) {
        // 跳过注释行
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        string key, value;
        if (iss >> key >> value) {
            if (key == "num_levels") {
                num_levels = stoi(value);
            } else if (key == "generate_density") {
                generate_density = (value == "true" || value == "1");
            } else if (key == "r_max") {
                r_max = stod(value);
            } else if (key == "r_points") {
                r_points = stoi(value);
            }
        }
    }
    
    infile.close();
}

int main() {
    // 默认参数
    int num_levels = 5;
    bool generate_density = false;
    double r_max = 20.0;
    int r_points = 1000;
    
    // 读取输入文件
    readInputFile("INPUT", num_levels, generate_density, r_max, r_points);
    
    // 创建氢原子实例
    Hydrogen hydrogen(num_levels, generate_density, r_max, r_points);
    
    // 求解氢原子能级
    hydrogen.solve();
    
    return 0;
}
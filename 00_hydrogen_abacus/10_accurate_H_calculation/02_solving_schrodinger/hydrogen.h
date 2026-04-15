#ifndef HYDROGEN_H
#define HYDROGEN_H

// 定义数学常量
#define _USE_MATH_DEFINES

#include <vector>
#include <string>
#include <cmath>

class Hydrogen {
private:
    int num_levels;        // 能级个数
    bool generate_density; // 是否生成电子密度
    double r_max;          // 径向最大距离
    int r_points;          // 径向网格点数
    double dr;             // 径向网格步长

public:
    // 构造函数
    Hydrogen(int num_levels, bool generate_density, 
             double r_max = 20.0, int r_points = 1000);

    // 求解氢原子能级
    void solve();

    // 生成电子密度文件
    void generateDensityFile(const std::string& filename);

private:
    // 求解径向方程
    double solveRadialEquation(int n, int l);

    // 求解角向方程（球谐函数）
    double sphericalHarmonic(int l, int m, double theta, double phi);

    // 计算径向波函数
    double radialWavefunction(int n, int l, double r);

    // 计算电子密度
    double electronDensity(int n, int l, int m, double r, double theta, double phi);

    // 计算连带勒让德多项式
    double associatedLegendre(int l, int m, double x);

    // 计算拉盖尔多项式
    double laguerrePolynomial(int n, int k, double x);
};

#endif // HYDROGEN_H
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
    double theta_max;      // 角度最大范围（弧度）
    int theta_points;      // 角度网格点数
    double phi_max;        // 方位角最大范围（弧度）
    int phi_points;        // 方位角网格点数

public:
    // 构造函数
    Hydrogen(int num_levels, bool generate_density, 
             double r_max = 20.0, int r_points = 1000,
             double theta_max = M_PI, int theta_points = 100,
             double phi_max = 2 * M_PI, int phi_points = 100);

    // 求解本征值和本征函数
    void solve();

    // 生成电子密度文件
    void generateDensityFile(const std::string& filename);

private:
    // 计算径向波函数 R(n, l, r)
    double radialWavefunction(int n, int l, double r);

    // 计算球谐函数 Y(l, m, theta, phi)
    double sphericalHarmonic(int l, int m, double theta, double phi);

    // 计算电子密度
    double electronDensity(int n, int l, int m, double r, double theta, double phi);

    // 计算连带勒让德多项式 P_l^m(cos(theta))
    double associatedLegendre(int l, int m, double x);

    // 计算拉盖尔多项式 L_n^k(x)
    double laguerrePolynomial(int n, int k, double x);
};

#endif // HYDROGEN_H
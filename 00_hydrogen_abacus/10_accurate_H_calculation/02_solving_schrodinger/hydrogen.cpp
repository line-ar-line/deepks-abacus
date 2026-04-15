#include "hydrogen.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// 玻尔半径（Å）
const double a0 = 0.529;
// 能量单位转换（ hartree 到 eV ）
const double hartree_to_ev = 27.2114;

// 构造函数
Hydrogen::Hydrogen(int num_levels, bool generate_density, 
                   double r_max, int r_points)
    : num_levels(num_levels), generate_density(generate_density),
      r_max(r_max), r_points(r_points) {
    dr = r_max / r_points;
}

// 求解氢原子能级
void Hydrogen::solve() {
    cout << "通过分别计算径向方程和角向方程求解氢原子能级..." << endl;
    
    int count = 0;
    for (int n = 1; count < num_levels; n++) {
        for (int l = 0; l < n; l++) {
            for (int m = -l; m <= l; m++) {
                if (count >= num_levels) break;
                
                // 求解径向方程得到能量
                double energy = solveRadialEquation(n, l);
                
                cout << "能级 " << count + 1 << ": n=" << n << ", l=" << l << ", m=" << m << ", E=" << energy << " eV" << endl;
                count++;
            }
        }
    }
    
    // 如果需要生成电子密度文件
    if (generate_density) {
        generateDensityFile("density.dat");
    }
}

// 求解径向方程
double Hydrogen::solveRadialEquation(int n, int l) {
    // 氢原子径向方程的解析解
    // 能量本征值只与主量子数n有关
    double energy = -13.6 / (n * n);
    return energy;
}

// 求解角向方程（球谐函数）
double Hydrogen::sphericalHarmonic(int l, int m, double theta, double phi) {
    double norm = sqrt((2 * l + 1) * tgamma(l - abs(m) + 1) / (4 * M_PI * tgamma(l + abs(m) + 1)));
    double legendre = associatedLegendre(l, abs(m), cos(theta));
    double harmonic = norm * legendre * cos(m * phi); // 只取实部
    return harmonic;
}

// 计算径向波函数
double Hydrogen::radialWavefunction(int n, int l, double r) {
    double rho = 2 * r / (n * a0);
    double norm = sqrt(pow(2.0 / (n * a0), 3) * tgamma(n - l) / (2 * n * tgamma(n + l + 1)));
    double laguerre = laguerrePolynomial(n - l - 1, 2 * l + 1, rho);
    double radial = norm * exp(-rho / 2) * pow(rho, l) * laguerre;
    return radial;
}

// 计算电子密度
double Hydrogen::electronDensity(int n, int l, int m, double r, double theta, double phi) {
    double radial = radialWavefunction(n, l, r);
    double harmonic = sphericalHarmonic(l, m, theta, phi);
    double density = radial * radial * harmonic * harmonic;
    return density;
}

// 计算连带勒让德多项式 P_l^m(x)
double Hydrogen::associatedLegendre(int l, int m, double x) {
    if (m < 0) {
        m = -m;
        double sign = (m % 2 == 0) ? 1.0 : -1.0;
        return sign * associatedLegendre(l, m, x);
    }
    
    // 递归计算
    if (l == m) {
        return pow(-1, m) * pow(1 - x * x, m / 2.0) * tgamma(2 * m + 1) / (pow(2, m) * tgamma(m + 1));
    } else if (l == m + 1) {
        return x * (2 * m + 1) * associatedLegendre(m, m, x);
    } else {
        return ( (2 * l - 1) * x * associatedLegendre(l - 1, m, x) - (l + m - 1) * associatedLegendre(l - 2, m, x) ) / (l - m);
    }
}

// 计算拉盖尔多项式 L_n^k(x)
double Hydrogen::laguerrePolynomial(int n, int k, double x) {
    if (n == 0) {
        return 1.0;
    } else if (n == 1) {
        return 1 + k - x;
    } else {
        return ( (2 * n + k - 1 - x) * laguerrePolynomial(n - 1, k, x) - (n + k - 1) * laguerrePolynomial(n - 2, k, x) ) / n;
    }
}

// 生成电子密度文件
void Hydrogen::generateDensityFile(const string& filename) {
    cout << "生成电子密度文件..." << endl;
    
    ofstream outfile(filename);
    if (!outfile) {
        cerr << "无法打开文件 " << filename << endl;
        return;
    }
    
    // 写入文件头
    outfile << "# 氢原子电子密度文件" << endl;
    outfile << "# 格式: r(Å) theta(rad) phi(rad) density(electrons/Å³)" << endl;
    
    // 计算并写入电子密度
    // 这里使用基态(n=1, l=0, m=0)的电子密度
    int theta_points = 50;
    int phi_points = 50;
    double dtheta = M_PI / theta_points;
    double dphi = 2 * M_PI / phi_points;
    
    for (int i = 0; i < r_points; i++) {
        double r = (i + 1) * dr;
        for (int j = 0; j < theta_points; j++) {
            double theta = j * dtheta;
            for (int k = 0; k < phi_points; k++) {
                double phi = k * dphi;
                
                double density = electronDensity(1, 0, 0, r, theta, phi);
                outfile << r << " " << theta << " " << phi << " " << density << endl;
            }
        }
    }
    
    outfile.close();
    cout << "电子密度文件已生成: " << filename << endl;
}
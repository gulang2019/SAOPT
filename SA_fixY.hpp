#include <vector>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <utility>
#include <cmath>
#include <iomanip>
#include <string>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
using namespace autodiff;
class SA{
    public:
    const int n_level;
    const int n_core;
    const int n_loop;
    const int remain_level;
    const int remain_loop;
    const std::vector<int> fixed;
    int * Cap;
    int * Size;
    int * BW;
    int * zeroInit;
    int p_level;
    int * p_num;
    const int fixed_idx;
    SA(
        int level,
        int core,
        int loop,
        int * capa,
        int * size, 
        int * bw, 
        int * zero,
        int *p_para,
        int p,
        std::vector<int> V,
        int fixed_i
    );
    bool neighbor(
        const VectorXreal x,
        VectorXreal & v,
        real (*f)(const VectorXreal &, const int *, const SA*),
        bool only_up = false
    );
    bool Up(const int idx, VectorXreal & y);
    bool Down(const int idx, VectorXreal & y);
    bool startingPoint(VectorXreal & x);
    template<int mod>
    real solve_unit(VectorXreal & v, int maxIter=200, int Tmax = 10000, int Tmin = 100, bool only_up = false, bool init = false);
    VectorXreal solve(VectorXreal & V);
    VectorXreal parallel_solve(VectorXreal & V, int * parallel);
    VectorXreal numAB_solve(VectorXreal & V, int * parallel, int * numAB);
    VectorXreal eval_grad_f(
        const VectorXreal x,
        real (*f)(const VectorXreal &, const int *,  const SA*) 
    );
    VectorXreal integrated_solver(std::string path = "./schedule/");
};
template<int mod>
real objectiveFunction(
    const VectorXreal & x, const int * p, const SA* t
);
VectorXreal constraintFunction(
    const VectorXreal & x, const SA * t
);
SA::SA(
    int level,
    int core,
    int loop,
    int * capa,
    int * size, 
    int * bw, 
    int * zero,
    int *p_para,
    int p,
    std::vector<int> V,
    int fixed_i
):n_core(core), n_level(level), n_loop(loop), p_level(p), remain_level(level - 1), remain_loop(loop - 3), fixed(V), fixed_idx(fixed_i)
{
    srand(time(NULL));
    Cap = new int [n_level - 1]; // register level is ignored. 
    Size = new int [n_loop];
    BW = new int [n_level];
    zeroInit = new int[n_loop];
    p_num = new int [3];
    for(int i = 0; i < n_level - 1; i++)
        Cap[i] = capa[i];
    for(int i = 0; i < n_loop; i++)
        Size[i] = size[i]; 
    for(int i = 0; i < n_level; i++)
        BW[i] = bw[i];
    for(int i = 0; i < n_loop; i++)
        zeroInit[i] = zero[i];
    for(int i = 0; i < 3; i++)
        p_num[i] = p_para[i];
}
bool SA::startingPoint(VectorXreal & x){
    std::cout << "begin starting point" << std::endl;
    int max_iter = 1;
    int curBest = 1e7;
    for(int i = 0; i < 3; i++)
        if (zeroInit[i] * p_num[i+1] > Size[i+1])
            return false;
    if (fixed_idx > 0 && (fixed[p_level - 1] * p_num[fixed_idx] > fixed[p_level]))
        return false;
    for(int iter = 1; iter <= max_iter; iter++){
        // std:: cout << "flag1" << std::endl;
        int t[remain_level][remain_loop];
        if(fixed_idx > 0)
            for(int i = 0; i < remain_level; i++){
                t[i][fixed_idx] = fixed[i];
            }
        for(int i = 0; i < remain_loop; i++){
            if (i == fixed_idx) continue;
            int T[remain_level];
            for(int j = 0; j < remain_level; j++){
                // [zeroInit[i + 1], Size[i + 1]]
                T[j] = rand() % (Size[i + 1] - zeroInit[i + 1] + 1) + zeroInit[i + 1];
            }
            std::sort(T, T + remain_level);
            for(int j = 0; j < remain_level; j++)
                t[j][i] = T[j];
            // std::cout <<"flag1.1" << std::endl;
            if (i <= 2 && t[2][i] < t[1][i] * p_num[i]){
                // std::cout << t[2][i] << " " << t[1][i] << " " << p_num[i] << " " << zeroInit[i + 1] << std::endl;
                int temp = 0;
                while(t[2][i] < t[1][i] * p_num[i + 1]){
                    temp ++;
                    if (temp % 2 == 0){
                        if (t[2][i] + 1 <= Size[i + 1])
                            t[2][i] += 1;
                    }
                    else{
                        if (t[1][i] - 1 >= zeroInit[i + 1])
                            t[1][i] -= 1;
                    }
                }
                t[0][i] = rand()%(t[1][i] - zeroInit[i +1] + 1) + zeroInit[i + 1];
            }
        }
        for(int i = 0; i < remain_level; i++){
            for(int j = 0; j < remain_loop; j++)
                std::cout << t[i][j] << " ";
            std::cout << std::endl; 
        }
        // std:: cout << "flag2" << std::endl;
        for(int i = 0; i < remain_level; i++){
            while(true){
                int * T = t[i];
                int v =  T[0] * T[3] * Size[5] * Size[6] + 
                1 * T[3] * (T[1] + Size[5] - 1) * (T[2] + Size[6] - 1) + 
                1 * T[0] * T[1] * T[2];
                std::cout << v << " "<< Cap[i] << std:: endl;
                if (v > Cap[i]){
                    for(int j = 0; j < remain_loop; j++){
                        if(j == fixed_idx) continue;
                        int low;
                        if(i == 0)  low = zeroInit[j + 1];
                        else  low = t[i - 1][j];
                        int high = T[j];
                        if(i == 2 && j <= 2) high /= p_num[j];
                        T[j] = rand() % (high - low + 1) + low; // uniform choose from [low, T[j]]
                    }
                }
                else { 
                break;}
            }
        }
        // std:: cout << "flag3" << std::endl;
        /*
        for(int i = remain_level - 1; i >= 0; i--){ 
            int * T = t[i];
            int Max;
            if (i == remain_level - 1){
                Max = Size[1] * Size[4] * Size[5] * Size[6] + 
                1 * Size[4] * (Size[2] + Size[5] - 1) * (Size[3] + Size[6] - 1) + 
                1 * Size[1] * Size[2] * Size[3];
            }
            else{
                int * T_next = t[i + 1];
                Max = T_next[0] * T_next[3] * Size[5] * Size[6] + 
                1 * T_next[3] * (T_next[1] + Size[5] - 1) * (T_next[2] + Size[6] - 1) + 
                1 * T_next[0] * T_next[1] * T_next[2];
            }
            if(Max <= Cap[i]){
                if(i == remain_level - 1){
                    for(int j = 0; j < remain_loop; j++){
                        if (j != fixed_idx)
                            T[j] = Size[j + 1];
                    }
                }
                else{
                    int * T_next = t[i + 1];
                    for(int j = 0; j < remain_loop; j++){
                        if (j != fixed_idx)
                            T[j] = T_next[j];
                    }
                }
                continue;
            }
            while(true){
                int v =  T[0] * T[3] * Size[5] * Size[6] + 
                1 * T[3] * (T[1] + Size[5] - 1) * (T[2] + Size[6] - 1) + 
                1 * T[0] * T[1] * T[2];
                if (v > Cap[i] * 0.1) {
                //std :: cout << "v, Cap[i]:";
                //std::cout << v << " " << Cap[i];
                    break;
                }
                int temp[remain_loop];
                for(int j = 0; j < remain_loop; j++){
                    if (j == fixed_idx) continue;
                    int high;
                    if(i == remain_level - 1) high = Size[j + 1];
                    else high = t[i + 1][j];
                    int low = T[j];
                    if (i == 2 && j <= 2)
                        low *= p_num[j];
                    if(low > high) 
                        continue;
                    temp[j] = rand() % (high - low + 1) + low;
                }
                v = temp[0] * temp[3] * Size[5] * Size[6] + 
                1 * temp[3] * (temp[1] + Size[5] - 1) * (temp[2] + Size[6] - 1) + 
                1 * temp[0] * temp[1] * temp[2];
                if(v > Cap[i]) continue;
                for(int j = 0; j < remain_loop; j++)
                    T[j] = temp[j];
            }
        }*/
        // std:: cout << "flag4" << std::endl;
        VectorXreal X(remain_level * remain_loop);
        for(int i = 0; i < remain_level; i++)
            for(int j = 0; j < remain_loop; j++){
                int prev;
                if(i == 0)
                    prev = zeroInit[j + 1];
                else prev = t[i - 1][j];
                if (j != fixed_idx)
                    t[i][j] = t[i][j] / prev * prev;
                X(i * remain_loop + j) = t[i][j];
            }
        real obj = objectiveFunction<0>(X, p_num, this);
        if (float(obj) < curBest){
            for(int i = 0; i < remain_level * remain_loop; i++)
                x[i] = int(X(i));
            curBest = float(obj);
        }
    }
    std::cout << "end starting point\n";
    return true; 
}

template<int mod>
real SA::solve_unit(VectorXreal & v, int maxIter, int Tmax, int Tmin, bool only_up, bool init){
    VectorXreal V(remain_level * remain_loop);
    real best = 1e8;
    std::ofstream out, Out;
    Out.open("Cap.txt", std::ios::app);
    out.open("Tiles.txt", std::ios::app);
    assert(V.size()== remain_level * remain_loop);
    if (init) V = v;
    else if (!startingPoint(V)){
        return -1;
    }
    else{
        v = V;
    }
    for(int i = 0; i < remain_level; i++){
        for(int j = 0; j < remain_loop; j++)
            out << V(i * remain_loop + j) << " ";
        out << std::endl;
    }
    real (*f)(const VectorXreal &,const int *,const SA *);
    f = objectiveFunction<mod>;
    real e = f(V, p_num, this);
    // std::cout << "begin solve 1.1" <<std::endl;
    // std::cout << e << std::endl;
    for(int T = Tmax; T >= Tmin; T /= 10)
    {
        real lastE = e;
        for(int i = 1; i <= maxIter; i ++){
            for(int ii = 0; ii < remain_level; ii++){
                for(int j = 0; j < remain_loop; j++)
                    out << V(ii * remain_loop + j) << " ";
                out << std::endl;
            }
            if (i % 20 == 0){
                if(-1e-5<=(e - lastE) / e&& (e - lastE) / e <= 1e-5)
                    break;
                else lastE = e;
            }
            VectorXreal tempV(remain_level * remain_loop);
            bool neighborFound = neighbor(V, tempV, f, only_up);
            if (!neighborFound){
                std::cout << "neighbor not found\n";
                break;
            }
            real tempE = f(tempV, p_num, this);
            float p = 1 / (1 + exp(- float(e - tempE) / T));
            // std::cout << p << std::endl;
            // std::cout << e << " " << tempE << " " << e - tempE << " " << p << std::endl;
            if ((double) rand() / RAND_MAX < p){
                V = tempV;
                e = tempE;
                out << std::endl;
                for(int i = 0; i < remain_level; i++){
                    for(int j = 0; j < remain_loop; j++)
                        out << V(i * remain_loop + j) << " ";
                    out << std::endl;
                }
                VectorXreal C = constraintFunction(V, this);
                for(int i = 0; i < remain_level; i++){
                    Out << C[i] / Cap[i] << " ";
                }
                Out << std::endl;
                // std::cout << e << std::endl;
            }
            if (tempE < best){
                best = tempE;
                v = V;
            }
            // std::cout << e << std::endl;
        }
    }
    Out.close();
    out.close();
    return best;
}

VectorXreal SA::solve(VectorXreal & V){
    VectorXreal e(n_level); 
    /*simulating anealing to minimize the argmax*/
    std::cout << "solve_unit 0" << std::endl;
    e(0) = solve_unit<0>(V, 200, 10000, 100, false, false);
    if (e(0) < 0){
        e(0) = 1e9;
        return e;
    }
    /*moutain climbing to minimize others*/
    std::cout << "solve_unit 1" << std::endl;
    e(1) = solve_unit<1>(V, 200, 1000, 100, true, true);
    std::cout << "solve_unit 2" << std::endl;
    e(2) = solve_unit<2>(V, 200, 1000, 100, true, true);
    std::cout << "solve_unit 3" << std::endl;
    e(3) = solve_unit<3>(V, 200, 1000, 100, true, true);
    std::cout << "end solve" << std::endl;
    return e;
}

bool cmp(std::pair<int, real> a, std::pair<int, real> b){
   return a.second > b.second;
}

template<int mod>
real objectiveFunction(
    const VectorXreal & x, const int * p, const SA* t
){
    // std::cout << "begin obejctive" << std::endl;
    const int n_level = t->n_level;
    const int n_loop = t->n_loop;
    const int n_core = t->n_core;
    const int remain_level = t->remain_level;
    const int remain_loop = t->remain_loop;
    const int p_level = t->p_level;
    VectorXreal output(n_level);
    real tile[n_level + 1][n_loop + 2];
    tile[0][0] = tile[0][n_loop + 1] = 1;
    for(int i = 1; i <= n_loop; i++)
        tile[0][i] = t->zeroInit[i - 1];
    for(int i = 1; i < n_level; i++){
        tile[i][0] = tile[i][n_loop + 1] = 1;
        tile[i][1] = 1;
        tile[i][6] = t->Size[5];
        tile[i][7] = t->Size[6];
        for(int j = 2; j < 2 + remain_loop; j++){
            tile[i][j] = x((i - 1) * remain_loop + j - 2);
        }
    }
    for(int j = 1; j < n_loop + 1; j++)
        tile[n_level][j] = t->Size[j - 1];   
    tile[n_level][0] = tile[n_level][n_loop + 1] = 1;
    std::ofstream out;
    out.open("objective.txt", std::ios::app);
    for(int i = 0; i <= n_level; i++){
        for(int j=  0; j < n_loop + 2; j++){
            out << tile[i][j] << " ";
        }
        out << std::endl;
    }
    for (int i = 0; i < n_level; i++)
    { 
        std::pair<int, real> seq[n_loop + 2];
        // float * T = Tile[i];
        real * T = tile[i];
        float value = 0;
        // b1 f2 x3 y4 c5 h6 w7
        seq[0] = std::make_pair(0,  (T[3] + T[6] - 1) * (T[4] + T[7] - 1) * T[1] * T[5] + 2 * T[1] * T[2] * T[3] * T[4] + T[2] * T[5] * T[6] * T[7]);
        seq[1] = std::make_pair(1, T[2] * T[5] * T[6] * T[7]);
        seq[2] = std::make_pair(2, (T[3] + T[6] - 1) * (T[4] + T[7] - 1) * T[1] * T[5]);
        seq[3] = std::make_pair(3, (T[6] - 1) * (T[4] + T[7] - 1)* T[1] * T[5] + T[2] * T[5] * T[6] * T[7]);
        seq[4] = std::make_pair(4,  (T[3] + T[6] - 1) * (T[7] - 1) * T[1] * T[5] + T[2] * T[5] * T[6] * T[7]);
        seq[5] = std::make_pair(5, 2 * T[1] * T[2] * T[3] * T[4]);
        seq[6] = std::make_pair(6,  (T[3] - 1) * T[1] * T[5] * (T[4] + T[7] - 1) + 2 * T[1] * T[2] * T[3] * T[4]);
        seq[7] = std::make_pair(7,  (T[4] - 1) * T[1] * T[5] * (T[3] + T[6] - 1)+2 * T[1] * T[2] * T[3] * T[4] );
        seq[8] = std::make_pair(8, 0);
        std::sort(seq, seq + n_loop + 2, cmp);
        real coeff = 1;
        if(i <= p_level)  coeff /= n_core;
        real p = 0;
        for (int j = 0; j <= n_loop + 1; j++)
            coeff = coeff/(T[seq[j].first]);
        for(int j = 0; j <= n_loop; j++){
            int idx = seq[j].first;
            coeff = coeff * (T[idx]) / (tile[i + 1][idx]);
            if (i == p_level && 2 <= idx && idx <= 4) coeff *= t->p_num[idx - 2];
            p = p + (seq[j].second - seq[j + 1].second)* coeff;
            out << idx << " ";
        } 
        out << std::endl;
        output(i) = p / t->BW[i];
    }
    out<<std::endl;
    out.close();
    out.open("argmax.txt", std::ios::app);
    int argmax = 0;
    for(int i = 0; i < n_level; i++){
        out << std::setw(20) << output(i);
        if(output(i) > output(argmax))
            argmax = i;
    }
    out << std::setw(20) << argmax<<std::endl;
    out.close();
    for(int i = 0; i < n_loop; i++)
        output *= t->Size[i];
    //std::cout << "end obj\n";
    //std::cout << std::endl;
    for(int i = 0; i < n_level; i++){
        for(int j = n_level - 1; j >= i + 1; j--){
            if (output(j) >= output(j - 1)){
                real temp = output(j);
                output(j) = output(j - 1);
                output(j - 1) = temp; 
            }
        }
    }
    // std::cout << "end obejctive" << std::endl;
    return output(mod);
    
}
int dec2(real r){
    int temp = float(r);
    int ret = 0;
    while(temp % 2 == 0){
        ret ++;
        temp /= 2;
    }
    return ret;
}
VectorXreal constraintFunction(
    const VectorXreal & x, const SA * t
){
    const int n_level = t->n_level;
    const int n_loop = t->n_loop;
    const int remain_level = t->remain_level;
    const int remain_loop = t->remain_loop;
    const int  p_level = t->p_level;
    VectorXreal output(remain_level);
    VectorXreal T(remain_loop);
    for(int i = 0; i < remain_level; i++){
        for(int j = 0; j < remain_loop; j++)
            T(j) = x(i * remain_loop + j);
        real r;
        int y_fac = std::min(
            4, std::min(
                dec2(T(2)), dec2(t->Size[3])
                )
            ), y_fac_exp = 1;
        for(int j = 0; j < y_fac; j++)   y_fac_exp *= 2;
        real y_length = 0;
        for(int j = 0; j < 16; j+= y_fac_exp)
            y_length += int((T(2) + 15 + j) / 16) * 16;
        y_length /= 16 / y_fac_exp;

        int yw_fac = std::min(
                        4, std::min(
                            dec2(T(2)), dec2(t->Size[6] + t->Size[3] - 1)
                            )
                        ), yw_fac_exp = 1;
        for(int j = 0; j < yw_fac; j++) yw_fac_exp *= 2;
        real yw_length = 0;
        for(int j = 0; j < 16; j += yw_fac_exp) 
            yw_length += int((T(2) + t->Size[6] + 14 + j) / 16) * 16;
        yw_length /= 16 / yw_fac_exp;
        // std::cout << yw_length << " " << T(2) + t->Size[6] - 1 << ";" << y_length << " " << T(2) << std::endl;

        output(i) = T(0) * T(3) * t->Size[5] * t->Size[6] + 
        T(3) * (T(1) + t->Size[5] - 1) * yw_length + 
        T(0) * T(1) * y_length;
    }
    return output; 
}
VectorXreal SA::eval_grad_f(
    const VectorXreal x, 
    real (*f)(const VectorXreal &, const int *,  const SA*)
)
{
   VectorXreal X(remain_level * remain_loop);
   X = x;
   real F;
   VectorXreal G = gradient(f, wrt(X), at(X, p_num, this), F);
   return G;
}

bool SA::Up(const int idx, VectorXreal &y){
    int loop_idx = idx % remain_loop;
    int level_idx = idx / remain_loop;
    int mult[remain_level - 1];
    for(int i = level_idx; i < remain_level - 1; i++)
        mult[i] = float(y[(i + 1) * remain_loop + loop_idx] / y[i * remain_loop + loop_idx]);
    int d;
    if (level_idx == 0)
        d = zeroInit[loop_idx + 1];
    else d = float(y[idx - remain_loop]);
    if (level_idx == 1 && loop_idx <= 2 && (y[idx] + d) * p_num[loop_idx] >= Size[loop_idx + 1]) return false;
    y[idx] += d;
    y[idx] = min(Size[loop_idx + 1], y[idx]);
    for(int i = level_idx; i < remain_level - 1; i++){
        int tmp = float(y[i * remain_loop + loop_idx]) * mult[i]; 
        tmp = std::min(tmp, Size[loop_idx + 1]);
        y[(i+1)*remain_loop + loop_idx] = tmp;
    }
    return true;
}

bool SA::Down(const int idx, VectorXreal &y){
    // std::cout << "begin Down" << std::endl;
    int loop_idx = idx % remain_loop;
    int level_idx = idx / remain_loop;
    int mult[remain_level - 1];
    for(int i = level_idx; i < remain_level - 1; i++)
        mult[i] = float(y[(i + 1) * remain_loop + loop_idx] / y[i * remain_loop + loop_idx]);
    int d;
    if (level_idx == 0)
        d = zeroInit[loop_idx + 1];
    else d = float(y[idx - remain_loop]);
    if (y[idx] <= d || (loop_idx <= 2 && level_idx == 2 && y[idx] < (p_num[idx] + 1) * d))
        return false;
    y[idx] -= d;
    for(int i = level_idx; i < remain_level - 1; i++){
        y[(i+1)*remain_loop + loop_idx] = y[i * remain_loop + loop_idx] * mult[i]; 
    }
    // std::cout << "end Down" << std::endl;
    return true;
}

bool cmp_n(std::pair<int, real> a, std::pair<int, real> b){
    return a.second < b.second;
}
bool SA::neighbor(
    const VectorXreal x,
    VectorXreal & V,  
    real (*f)(const VectorXreal &, const int *, const SA*),
    bool only_up
){
    VectorXreal v(remain_level * remain_loop);
    std::ofstream out;
    out.open("grad.txt", std::ios::app);
    VectorXreal grad_f = eval_grad_f(x, f);
    for(int i = 0; i < remain_level; i++){
        for(int j = 0; j < remain_loop; j++)
            out << grad_f(i*remain_loop + j) << " ";
        out << std::endl;
    }
    out<<std::endl;
    out.close();
    std::vector<std::pair<int, real>> temp(remain_level * remain_loop);
    for(int i = 0; i < remain_level * remain_loop; i++){
        temp[i] = std::make_pair(i, grad_f(i));
    }
    
    // std::sort(temp.begin(), temp.end(), cmp_n);
    real best = 1e8;
    bool bestFound = false;
    for(int i = 0; i < remain_level * remain_loop; i++)
    {
        // if(temp[i].second >= -1e-6) continue;
        // int inc_idx = temp[i].first;
        if(grad_f[i] >= -1e-6) continue;
        int inc_idx = i;
        if(inc_idx % remain_loop == fixed_idx) continue;
        if(x[inc_idx] >= Size[inc_idx % remain_loop + 1] - 0.5)
            continue;
        v = x;
        if(!Up(inc_idx, v)){
            if(inc_idx / remain_loop == 0) continue;
            else if(!Up(inc_idx, v)) continue;
        }
        // std::cout << inc_idx << std::endl;
        if (only_up){
            VectorXreal C = constraintFunction(v, this);
            bool flag = true;
            for(int j = 0; j < remain_level; j++)
                if(C[j] > Cap[j])
                    flag = false;
            if (!flag)
                continue;
            bestFound = true;
            V = v;
            break;
        }
        else{
            bool fail = false;
            for(int j = 0; j < remain_level; j++)
            {
                VectorXreal C = constraintFunction(v, this); 
                bool avail[4] = {true, true, true, true};
                avail[inc_idx % remain_loop] = false;
                avail[fixed_idx] = false;
                int cnt = 2 - (inc_idx % remain_loop == fixed_idx);
                while (C[j] > Cap[j]){
                    int dec_idx = j * remain_loop + rand() % remain_loop;
                    if (!avail[dec_idx % remain_loop]) continue;
                    if (!Down(dec_idx % remain_loop, v)){
                        cnt ++;
                        avail[dec_idx % remain_loop] = false;
                        if (cnt == 4){
                            fail = true;
                            break;
                        }
                    }
                    C = constraintFunction(v, this);
                }
                if(fail) break;
            }
            if(fail) continue;
            real obj = f(v, p_num, this);
            if (best > obj){
                bestFound = true;
                best = obj;
                V = v;
            }
        }
    }
    return bestFound;
}

VectorXreal SA::parallel_solve(VectorXreal& x, int * parallel){
    int depth = 0;
    int num = n_core;
    VectorXreal best(n_level);
    best << 1e9, 1e9, 1e9, 1e9;
    for(int i = 0; i < 3; i++)
        p_num[i] = 0;
    struct s{
        VectorXreal r; 
        VectorXreal X;
    }XX[5];
    while(true){
        if(depth == 2){
            p_num[depth] = num;
            num = 1;
            for(int i = 0; i < 3; i++){
                std::cout << p_num[i] << " ";
            }
            std::cout << std::endl;
            # pragma omp parallel for
            for(int iter = 0; iter < 5; iter ++){
                XX[iter].X = VectorXreal(remain_level * remain_loop);
                XX[iter].r = solve(XX[iter].X);
            }
            for(int iter = 0; iter < 5; iter ++){
                VectorXreal r = XX[iter].r;
                for(int i = 0; i < n_level; i++){
                        if(best(i) > r(i)){
                            best = r;
                            x = XX[iter].X;
                            for(int j = 0; j < 3; j++)
                                parallel[j] = p_num[j];
                            break;
                        }
                        else break;
                    }
            }
            while(num == 1 && depth >= 0){
                num *= p_num[depth];
                p_num[depth] = 0;
                depth --;
            }
        }
        if(depth == -1) break;
        if (p_num[depth] == 0){
            p_num[depth] = 1;
            depth ++;
            continue;
        }
        for(int i = p_num[depth] + 1; i <= num * p_num[depth]; i++){
            if ((p_num[depth] * num) % i == 0){
                num = num * p_num[depth] / i;
                p_num[depth] = i;
                depth ++;
                break;
            }
        }
        
    }   
    return best;
}
VectorXreal SA::numAB_solve(VectorXreal & V, int * parallel, int * numAB){
    VectorXreal v(remain_level * remain_loop);
    int para[3];
    int numAB_candidates[2][2] = {{16, 6}, {24, 4}};
    VectorXreal best(n_level);
    for(int i = 0; i < n_level; i++) best(i) = 1e8;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < n_level; j++)
            zeroInit[j] = 1;
        zeroInit[1] = numAB_candidates[i][0];
        zeroInit[3] = numAB_candidates[i][1];
        VectorXreal r = parallel_solve(v, para);
        for(int j = 0; j < n_level; j++){
            if(best(j) > r(j)){
                best = r;
                for(int k = 0; k < 3; k++)  parallel[k] = para[k];
                for(int k = 0; k < 2; k++) numAB[k] = numAB_candidates[i][k];
                V = v;
                break;
            }
            else break;
        }
    }
    return best;
}
VectorXreal SA::integrated_solver(std::string path){
    int parallel[3], numAB[2] = {16, 6};
    VectorXreal x(remain_level * remain_loop);
    VectorXreal result = parallel_solve(x, parallel);
    VectorXreal output(n_level);
    real tile[n_level + 1][n_loop + 2];
    tile[0][0] = tile[0][n_loop + 1] = 1;
    for(int i = 1; i <= n_loop; i++)
        tile[0][i] = 1;
    tile[0][2] = numAB[0];
    tile[0][4] = numAB[1];
    for(int i = 1; i < n_level; i++){
        tile[i][0] = tile[i][n_loop + 1] = 1;
        tile[i][1] = 1;
        tile[i][6] = Size[5];
        tile[i][7] = Size[6];
        for(int j = 2; j < 2 + remain_loop; j++){
            tile[i][j] = x((i - 1) * remain_loop + j - 2);
        }
    }
    for(int j = 1; j < n_loop + 1; j++)
        tile[n_level][j] = Size[j - 1];   
    tile[n_level][0] = tile[n_level][n_loop + 1] = 1;
    std::string filename = "";
    for(int i = 0; i < n_loop; i++){
        if(i != 0) filename+= '_';
        filename += std::to_string(Size[i]);
    }
    std::ofstream out;
    out.open(path + filename + ".txt",  std::ios::trunc);
    char tilename[10] = "_bfxychw_";
    out << "# tileSize\n";
    for (int i = 0; i < n_level; i++)
    { 
        std::pair<int, real> seq[n_loop + 2];
        // float * T = Tile[i];
        real * T = tile[i];
        float value = 0;
        // b1 f2 x3 y4 c5 h6 w7
        seq[0] = std::make_pair(0,  (T[3] + T[6] - 1) * (T[4] + T[7] - 1) * T[1] * T[5] + 2 * T[1] * T[2] * T[3] * T[4] + T[2] * T[5] * T[6] * T[7]);
        seq[1] = std::make_pair(1, T[2] * T[5] * T[6] * T[7]);
        seq[2] = std::make_pair(2, (T[3] + T[6] - 1) * (T[4] + T[7] - 1) * T[1] * T[5]);
        seq[3] = std::make_pair(3, (T[6] - 1) * (T[4] + T[7] - 1)* T[1] * T[5] + T[2] * T[5] * T[6] * T[7]);
        seq[4] = std::make_pair(4,  (T[3] + T[6] - 1) * (T[7] - 1) * T[1] * T[5] + T[2] * T[5] * T[6] * T[7]);
        seq[5] = std::make_pair(5, 2 * T[1] * T[2] * T[3] * T[4]);
        seq[6] = std::make_pair(6,  (T[3] - 1) * T[1] * T[5] * (T[4] + T[7] - 1) + 2 * T[1] * T[2] * T[3] * T[4]);
        seq[7] = std::make_pair(7,  (T[4] - 1) * T[1] * T[5] * (T[3] + T[6] - 1)+2 * T[1] * T[2] * T[3] * T[4] );
        seq[8] = std::make_pair(8, 0);
        std::sort(seq, seq + n_loop + 2, cmp);
        real coeff = 1;
        if(i <= p_level)  coeff /= n_core;
        real p = 0;
        for(int j = 1; j <= n_loop; j++){
            int idx = seq[j].first;
            out << tilename[idx] << " " << tile[i][idx] << ";";
        }
        out << std::endl;
        for (int j = 0; j <= n_loop + 1; j++)
            coeff = coeff/(T[seq[j].first]);
        for(int j = 0; j <= n_loop; j++){
            int idx = seq[j].first;
            coeff = coeff * (T[idx]) / (tile[i + 1][idx]);
            if (i == p_level && 2 <= idx && idx <= 4) coeff *= parallel[idx - 2];
            p = p + (seq[j].second - seq[j + 1].second)* coeff;
        } 
        output(i) = p;
    }
    for(int i = 0; i < n_loop; i++)
        output *= Size[i];
    out << "# prediction\n";
    for(int i = 0; i < n_level; i++)
        out << output(i) << " ";
    out << std::endl;
    out << "# parallel\n";
    for(int i = 0; i < 3; i++)
        out << parallel[i] << " ";
    out << std::endl;
    out << "# numAB\n";
    for(int i = 0; i < 2; i++)
        out << numAB[i] << " ";
    out << std::endl;
    out.close();
    return result;
}
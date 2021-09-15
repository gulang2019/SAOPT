#include "SA_fixY.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
int main(int argc, char**argv){
    std::cout << argc << std::endl;
    int nb = atoi(argv[1]);
    int nf = atoi(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    int nc = atoi(argv[5]);
    int nw = atoi(argv[6]);
    int nh = atoi(argv[7]);
    int strideX = atoi(argv[8]);
    int strideY = atoi(argv[9]);
    std::cout << "nb, " << nb << ", nf, " << nf << " nx " << nx << " ny " << ny << " nc "<< nc << " nw " << nw << " nh " << nh << "\n";
    int fixed_index = -1;
    std::vector<int> fixed_value;
    if(ny <= 50){
        fixed_index = 2;
        for(int i = 0; i < 3; i++)
            fixed_value.push_back(ny);
    }
    int level = 4;
    int loop = 7;
    int Size[7] = {nb,nf,nx,ny,nc,nw,nh};
    int Cap[3] = {32 * 1000 / 4, 256 * 1000 / 4, 25600 * 1000 / 4};
    int p_level = 2;
    int BW[4] = {46,37,30,11};
    std::string path;
    if(argc > 11){
        for(int i = 0; i < 4; i++)
            BW[i] = atoi(argv[10 + i]);
        path = argv[14];
    }
    else{
        path = argv[10];
    }
    std::cout << path << std::endl;
    int n_core = 2;
    int p_num[3] = {2,1,1};
    int zeroInit[7] = {1,16,1,6,1,1,1};
    float obj_scale = 0.01;
    int numAB[2];
    SA* solver = new SA(level, n_core, loop, Cap, Size, BW, zeroInit, p_num, p_level, fixed_value, fixed_index);
    VectorXreal X(3 * 4);
    VectorXreal r = solver->integrated_solver(path);
    /*
    for(int i = 0; i < 4; i++)
        std::cout << r(i) << " ";
    std::cout << std::endl;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 4; j++){
            std::cout << X(i * 4 + j) << " "; 
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for(int i = 0; i < 3; i++)
        std::cout << p_num[i] << " ";
    std::cout << std::endl;
    for(int i = 0; i < 2; i++)
        std::cout << numAB[i] << " ";
    std::cout << std::endl;
    std::cout << constraintFunction(X, solver);
    */
    return 0;
}

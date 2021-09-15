#include <iostream>

using namespace std;

int dec2(int r){
    int temp = float(r);
    int ret = 0;
    while(temp % 2 == 0){
        ret ++;
        temp /= 2;
    }
    return ret;
}
int main(){
    int Ny, Nw;
    cin >> Ny >> Nw;
    for(int Ty = 1; Ty <= Ny; Ty ++){
        int y_fac = min(
                4, min(
                    dec2(Ty), dec2(Ny)
                    )
                ), y_fac_exp = 1;
        for(int j = 0; j < y_fac; j++)   y_fac_exp *= 2;
        float y_length = 0;
        for(int j = 0; j < 16; j+= y_fac_exp)
            y_length += int((Ty + 15 + j) / 16) * 16;
        y_length /= 16.0 / y_fac_exp;
        cout << Ty << " " << y_length << " " << float(Ty) / y_length << endl;
    }
    return 0;
}
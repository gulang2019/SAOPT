## 改进后的cacheline更接近于真实情况下的cache
1. 可以把改进后的tensorsize加入constraintFunction
2. 先不改损失函数，保持datamovement的估计不变
- O = bfxy Tf * Tx * ([(Ty, Ny, 16)]_16 + Ty + 15) // 16) ) * 16 
- I = bc x+h-1 y+w-1 Tfc* (Tx + Th - 1) * ([(Ny + Nw - 1, Ty, Tw, 16)]_16 + 14 + Ty + Tw] // 16) * 16

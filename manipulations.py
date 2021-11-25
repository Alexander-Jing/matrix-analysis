import numpy as np
import argparse
import Gram_Schmidt
from Gram_Schmidt import Gram_Schmidt
import Givens
from Givens import Givens
import householder
from householder import householder
import PLU
from PLU import PLU 
import URV
from URV import URV
import argparse

if __name__ == "__main__":
    # 测试A矩阵，如果想要换别的矩阵的话，可以在这里修改
    A = np.array([
        [1,2,-3,4],
        [4,8,12,-8],
        [2,3,2,1],
        [-3,-1,1,-4]])
    
    QR = Gram_Schmidt(A)
    Q,R = QR.Shmidt_factor(A)
    QR.lin_equ(A,Q,R,b1=np.array([4,16,8,-7]))
    QR.det_calc(A,Q,R)

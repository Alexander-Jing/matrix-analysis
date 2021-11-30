import numpy as np
import argparse
from Gram_Schmidt import Gram_Schmidt
from Givens import Givens
from householder import housholder
from PLU import PLU 
from URV import URV

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--manipulation','-man',default="Householder",help="choose the manipulation,PLU,Gram-Schmidt,Householder,Givens,URV ")
    parser.add_argument('--A', action='store', type=float, nargs='+',help="the original matrix")
    parser.add_argument('--nrowsA', action='store', type=int,help="the number of rows of original matrix")
    parser.add_argument('--b',action='store', type=float, nargs='+',help="the original equation b (the form Ax=b)")
    parser.add_argument('--nrowsb', action='store', type=int,help="the number of rows of equation b (the form Ax=b)")
    
    orders = parser.parse_args()

    A = np.array(orders.A).reshape((orders.nrowsA, len(orders.A)//orders.nrowsA))
    b = np.array(orders.b) #.reshape((orders.nrowsb, len(orders.b)//orders.nrowsb))
    if str(orders.manipulation)=="Householder":
        HS = housholder(A)
        R,Q = HS.household_red(A)
        HS.lin_equ(A,R,Q,b1=b)
        HS.det_calc(A,R,Q)
    
    elif str(orders.manipulation)=="Gram-Schmidt":
        QR = Gram_Schmidt(A)
        Q,R = QR.Shmidt_factor(A)
        QR.lin_equ(A,Q,R,b1=b)
        QR.det_calc(A,Q,R)

    elif str(orders.manipulation)=="PLU":
        plu = PLU(A,b)
    elif str(orders.manipulation)=="Givens":
        GI = Givens(A)
        R,Q = GI.givens_reduction(A)
        GI.lin_equ(A,R,Q,b1=b)
        GI.det_calc(A,R,Q)
    elif str(orders.manipulation)=="URV":
        urv = URV(A,b)
        urv.URV_factor(A)
        U,V,C = urv.SVD_factor(A)
        urv.lin_equ(A,U,C,V,b)
        urv.det_calc(A,U,C,V) 
    else:
        print("The manipulation should be one of the follows: PLU,Gram-Schmidt,Householder,Givens,URV")


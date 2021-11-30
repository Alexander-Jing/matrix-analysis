import numpy as np
import argparse

class Givens():
    """the Givens reduction
    the Givens reduction of a nonsingular matrix

    input:
        A(np.array): the original matrix 
        b(np.array): Ax=b, b in the system of linear equations, in the form of columns
    Attributes:
        self.A(np.array): the original matrix
        self.b(np.array): Ax=b, b in the system of linear equations  
    methods: 
        def household_red()

    """
    def __init__(self,A,b=np.array([4,16,8,-7])):
        self.A = A
        self.b = b
    
    def givens_reduction(self,A0):
        """givens reduction 
        the givens reduction of a nonsingular matrix

        Args:
            A0(np.array): the input matrix 
        
        returns:
            R(np.array): the orthonormalization R matrix, used for\
                elementary reflector
            Q(np.array): the reduction result Q matrix

        """
        R = []
        A = A0.copy()
        n,_ = A.shape
        R0 = np.identity(n)
        
        for i in range(n-1):
            for j in range(i+1,n):
                xi = A[i,i]
                xj = A[j,i]
                c = xi/(np.sqrt(xi**2+xj**2))
                s = xj/(np.sqrt(xi**2+xj**2))
                R0[i,i] = c;R0[j,j] = c
                R0[j,i] = -s;R0[i,j] = s  # form the orthonormalization matrix
                A = np.dot(R0,A)
                R.append(R0)
                R0 = np.identity(n) 
        R = R[::-1]
        R = np.linalg.multi_dot(R)
        Q = np.dot(R,A0)  # calculate the Q matrix
        np.set_printoptions(suppress=True)
        print("****manipulation givens_reduction****")
        print("A","R","Q");print(A0,"\n",R,"\n",Q)
        print("A=R.T Q",", I = R.T R");print(np.dot(R.T,Q));print(np.dot(R.T,R))
        
        return R,Q

    def lin_equ(self,A,R,Q,b1):
        """get the solution of the system of linear equations 

        Args:
            A(np.array): the input matrix 
            R(np.array): the orthonormalization R matrix, used for\
                elementary reflector
            Q(np.array): the reduction result Q matrix  
            b1(np.array): Ax=b, b in the system of linear equations, in the form of columns

        returns:
            x(np.array): the solution of the system of linear equations 
        """
        n,_ = A.shape
        b = np.dot(R,b1.reshape([n,1]))  # RAx = Rb, Qx = Rb
        b = b.T[0]  
        x = []
        for k in range(n):
            if k==0:
                x.append(b[n-1-k]/Q[n-1-k,n-1-k])
            else:    
                x.append((b[n-1-k]-np.dot(Q[n-1-k,n-1-k+1:],(np.array(x[::-1])).T))/Q[n-1-k,n-1-k])
        x =np.array(x[::-1])
        print("the nonsingular system")
        print(A,b1)
        print("the solutions:")
        print(x)
        print("Ax")
        print(np.dot(A,x))
        return x
    
    def det_calc(self,A,R,Q):
        """calculate the determinant 
        Args:
            A(np.array): the input matrix 
            R(np.array): the orthonormalization R matrix, used for\
                elementary reflector
            Q(np.array): the reduction result Q matrix   
        
        returns:
            value(float): the absolute value of the determinant\
                take care, it's hard to know the correct value of the determinant\
                    because det(Q) = +-1, 
        """
        n,_ = A.shape
        detQ = 1
        for i in range(n):
            detQ = detQ*Q[i,i]
        detQ = np.abs(detQ)
        detA = detQ*1
        print("det(A) absolute value")
        print(detA)
        print("validate the det(A)")
        print(np.linalg.det(A))
        
        return detA


if __name__ == "__main__":
    # 测试A矩阵，如果想要换别的矩阵的话，可以在这里修改
    A = np.array([
        [1,2,-3,4],
        [4,8,12,-8],
        [2,3,2,1],
        [-3,-1,1,-4]])
    b1=np.array([4,16,8,-7])
    GI = Givens(A)
    R,Q = GI.givens_reduction(A)
    GI.lin_equ(A,R,Q,b1=np.array([4,16,8,-7]))
    GI.det_calc(A,R,Q)

                 
import numpy as np
import argparse

from numpy.linalg.linalg import det

class housholder():
    """the householder reduction  
    
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
    
    def household_red(self,A0):
        """householder reduction 
        the householder reduction of a nonsingular matrix

        Args:
            A(np.array): the input matrix 
        
        returns:
            R(np.array): the Shmidt orthonormalization R matrix, used for\
                elementary reflector
            Q(np.array): the reduction result Q matrix 

        """
        R = []
        A = A0.copy()
        n,_ = A.shape
        R0 = np.identity(n)

        for i in range(n-1):
            e = np.array([0.0 for _ in range(n-i)])
            e[0] = 1.0
            u = A[i:n,i] + np.linalg.norm(A[i:n,i])*e.T   # the u vector
            r = np.identity(n-i) - (2/np.dot(u.reshape([1,n-i]),u.reshape([n-i,1])))*np.dot(u.reshape([n-i,1]),u.reshape([1,n-i]))  # the R matrix
            R0[i:n,i:n] = r
            R.append(R0)
            A = np.dot(R0,A)
            R0 = np.identity(n)

        R = R[::-1]
        R = np.linalg.multi_dot(R)
        Q = np.dot(R,A0)
        np.set_printoptions(suppress=True)
        print("****manipulation household_red****")
        print("A","R","Q");print(A0,"\n",R,"\n",Q)
        print("A=R.T Q",", I = R.T R");print(np.dot(R.T,Q));print(np.dot(R.T,R))

        return R,Q
    
    def lin_equ(self,A,R,Q,b1):
        """get the solution of the system of linear equations 

        Args:
            A(np.array): the input matrix 
            R(np.array): the Shmidt orthonormalization R matrix, used for\
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
            R(np.array): the Shmidt orthonormalization R matrix, used for\
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
        detA = np.abs(detQ*1)
        print("det(A) absolute value")
        print(detA)
        print("validate the det(A)")
        print(np.linalg.det(A))
        
        return A


"""if __name__ == "__main__":
    # 测试A矩阵，如果想要换别的矩阵的话，可以在这里修改
    A = np.array([
        [1,2,-3,4],
        [4,8,12,-8],
        [2,3,2,1],
        [-3,-1,1,-4]])
    HS = housholder(A)
    R,Q = HS.household_red(A)
    HS.lin_equ(A,R,Q,b1=np.array([4,16,8,-7]))
    HS.det_calc(A,R,Q)"""
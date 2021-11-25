from warnings import simplefilter
import numpy as np
import Gram_Schmidt
from Gram_Schmidt import Gram_Schmidt
import argparse

class URV(Gram_Schmidt):
    """the URV factorization, inherited from the Gram_Schmidt class
    the URV factorization of a nonsingular matrix, take care \
        this class only fits the situation that the matrix is the nonsingular matrix (the RPN nonsingular matrix)

    input:
        A(np.array): the original matrix 
        b(np.array): Ax=b, b in the system of linear equations, in the form of columns
    Attributes:
        self.A(np.array): the original matrix
        self.b(np.array): Ax=b, b in the system of linear equations  
    methods: 
        def URV_factor(self,A0):
    
    """
    def __init__(self,A,b=np.array([4,16,8,-7])):
        super(Gram_Schmidt,self).__init__()
        self.A = A
        self.b = b
        
        
    
    def URV_factor(self,A0):
        """URV factorization 
        the URV factorization of a nonsingular matrix

        Args:
            A0(np.array): the input matrix 
        
        returns:
            U(np.array): the orthonormalization U matrix, the orthogonal basis of R(A)
            C(np.array): the nonsingular matrix, U.T A U = C
            
        """ 
        U,_ = self.Shmidt_factor(A0)
        C = np.linalg.multi_dot([U.T, A0, U])
        print("****manipulation URV_factor ****")
        print("A","U","C");print(A0,"\n",U,"\n",C)
        print("U C U.T",", I = U.T U");print(np.linalg.multi_dot([U, C, U.T]));print(np.dot(U.T,U))

        return U,C
        

if __name__ == "__main__":
    # 测试A矩阵，如果想要换别的矩阵的话，可以在这里修改
    A = np.array([
        [1,2,-3,4],
        [4,8,12,-8],
        [2,3,2,1],
        [-3,-1,1,-4]])
    b=np.array([4,16,8,-7])
    urv = URV(A,b)
    U,C = urv.URV_factor(A)
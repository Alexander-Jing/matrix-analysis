import numpy as np
from numpy.lib.function_base import copy
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

    def SVD_factor(self,A0):
        """SVD factorization 
        the SVD factorization of a nonsingular matrix

        Args:
            A0(np.array): the input matrix 
        
        returns:
            U(np.array): the orthogonal U matrix, the eigenvector of A A.T
            V(np.array): the orthogonal V matrix, the eigenvector of A.T A
            C(np.array): the orthogonal C matrix, the eigenvector of A.T A
            
        """ 
        A = A0.copy()
        n,_ = A.shape
        # calculate the eigenvalue and the eigenvector of the A A.T
        eigen_AAT = np.linalg.eigh(np.dot(A,A.T))
        eigen_val_AAT = np.array(eigen_AAT[0].tolist()[::-1])  # arrange the eigenvalues
        eigen_vec_AAT = np.array((eigen_AAT[1].T).tolist()[::-1]).T  # arrange the eigenvectors
        U = eigen_vec_AAT #the eigenvectors has already been orthogonal, so form the U matrix 
        # calculate the eigenvector of the A.T A
        eigen_ATA = np.linalg.eigh(np.dot(A.T,A))
        eigen_vec_ATA = np.array((eigen_ATA[1].T).tolist()[::-1]).T  # arrange the eigenvectors
        V = eigen_vec_ATA # the eigenvectors has already been orthogonal, so form the V matrix 
        
        for i in range(n):
            if (np.dot(A,V)[:,i][0]/U[:,i][0]<0):
                # take care, in fact the element of the C should be larger then 0, so Av/u > 0, this condition must be satisfied 
                U[:,i] = (-1)* U[:,i]
        
        # form the C matrix 
        C = np.zeros((n,n))
        for i in range(n):
            C[i,i] = np.sqrt(eigen_val_AAT[i])
        
        print("****manipulation SVD****")
        print("A","U","V","C");print(A0,"\n",U,"\n",V,"\n",C)
        print("A=U C V.T");print(np.linalg.multi_dot([U,C,V.T]))
        
        return U,V,C
    
    def lin_equ(self,A,U,C,V,b1):
        """get the solution of the system of linear equations 

        Args:
            A(np.array): the input matrix 
            U(np.array),V(np.array),C(np.array): the result of the svd
            b(np.array): Ax=b, b in the system of linear equations, in the form of columns

        returns:
            x(np.array): the solution of the system of linear equations 
        """
        n,_ = A.shape
        C_inv = C.copy()
        for i in range(n):
            C_inv[i,i] = 1/C_inv[i,i]
        x = np.linalg.multi_dot([V,C_inv,U.T,b1])
        print("the nonsingular system")
        print(A,b1)
        print("the solutions:")
        print(x)
        print("Ax")
        print(np.dot(A,x))
        return x

    def det_calc(self,A,U,C,V):
        """calculate the determinant 
        Args:
            A(np.array): the input matrix 
            U(np.array),V(np.array),C(np.array): the result of the svd
            b(np.array): Ax=b, b in the system of linear equations, in the form of columns
        
        returns:
            value(float): the absolute value of the determinant\
                take care, it's hard to know the correct value of the determinant\
                    because det(Q) = +-1, 
        """
        n,_ = A.shape
        detC = 1
        
        for i in range(n):
            detC = detC*C[i,i]
        detA = np.abs(detC*1)
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
    b=np.array([4,16,8,-7])
    urv = URV(A,b)
    urv.URV_factor(A)
    U,V,C = urv.SVD_factor(A)
    urv.lin_equ(A,U,C,V,b)
    urv.det_calc(A,U,C,V)

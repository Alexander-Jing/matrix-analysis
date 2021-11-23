import numpy as np
import argparse

class Gram_Schmidt():
    """the Shmidt factorization of the nonsingular matrix
    the Shmidt orthonormalization of the matrix(or the set of independent vectors)
    
    input:
        A(np.array): the original matrix 
        b(np.array): Ax=b, b in the system of linear equations, in the form of columns
    Attributes:
        self.A(np.array): the original matrix
        self.b(np.array): Ax=b, b in the system of linear equations  

    """
    def __init__(self,A,b=np.array([4,16,8,-7])):
        self.A = A
        self.b = b
    
    def Shmidt_factor(self,A):
        """the Shmidt factorization
        make the Shmidt orthonormalization and factorization 

        Args:
            A(np.array): the input matrix 
        
        returns:
            Q(np.array): the Shmidt orthonormalization Q matrix
            R(np.array): the Shmidt factorization R matrix 
        
        """
        # initialize the Q R matrix 
        Q = []
        R = []
        n,_ = A.shape  
        v0 = np.array([0.0 for _ in range(n)])

        # calculate the first column of the Q R matrix
        v = np.linalg.norm(A[:,0])
        Q.append(A[:,0]/v)
        v0[0] = v
        R.append(v0.tolist())
        
        # calculate the other columns
        for i in range(1,n):
            # calculate the q,v (q|ai)
            a = A[:,i]
            q = a - np.linalg.multi_dot([np.array(Q).T,np.array(Q),a])  # ai - Q Q.T ai
            v  = np.linalg.norm(q)
            v0[i] = v
            v0[0:i] = np.dot(np.array(Q),a).T
            
            Q.append(q/v)
            R.append(v0.tolist())
        
        Q = np.array(Q).T
        R = np.array(R).T

        np.set_printoptions(suppress=True)
        print("****manipulation****")
        print("A","Q","R");print(A,"\n",Q,"\n",R)
        print("QR","Q.T Q");print(np.dot(Q,R));print(np.dot(Q,Q.T))
        
        return Q,R
    
    def lin_equ(self,A,Q,R,b1):
        """get the solution of the system of linear equations 

        Args:
            A(np.array): the input matrix 
            Q(np.array): the Shmidt orthonormalization Q matrix
            R(np.array): the Shmidt factorization R matrix 
            b(np.array): Ax=b, b in the system of linear equations, in the form of columns

        
        returns:
            x(np.array): the solution of the system of linear equations 
        """
        n,_ = A.shape
        b = np.dot(Q.T,b1.reshape([n,1]))  # Rx = Q.T b
        b = b.T[0]  
        x = []
        
        for k in range(n):
            if k==0:
                x.append(b[n-1-k]/R[n-1-k,n-1-k])
            else:    
                x.append((b[n-1-k]-np.dot(R[n-1-k,n-1-k+1:],(np.array(x[::-1])).T))/R[n-1-k,n-1-k])
        x =np.array(x[::-1])
        print("the nonsingular system")
        print(A,b1)
        print("the solutions:")
        print(x)
        print("Ax")
        print(np.dot(A,x))
        return x
    
    def det_calc(self,A,Q,R):
        """calculate the determinant 
        Args:
            A(np.array): the input matrix 
            Q(np.array): the Shmidt orthonormalization Q matrix
            R(np.array): the Shmidt factorization R matrix 
        
        returns:
            value(float): the absolute value of the determinant\
                take care, it's hard to know the correct value of the determinant\
                    because det(Q) = +-1, 
        """
        n,_ = A.shape
        detR = 1
        for i in range(n):
            detR = detR*R[i,i]
        detA = detR*1
        print("det(A)")
        print(detA)
        print("validate the det(A)")
        print(np.linalg.det(A))
        
        return A



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

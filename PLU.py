import numpy as np

class PLU():
    def __init__(self,A,b=np.array([4,16,8,-7])):
        self.A = A
        self.Pi = []
        self.Gi = []
        self.Gi_inv = [] 
        # 这用类似于行列操作的方法直接进行变换
        L,U,P = self.PLU_trans()
        # 求解方程组
        self.sol_calc(L,U,P,b)
        self.det_calc(A,L,U,P)


    def P_i(self,A,ind):
        """这部分计算每次需要的置换矩阵(第一类初等变换)Pi，对于A中主元的提取
        参数：
            A(np array)： 输入的变换的矩阵
            ind(int): 选择行
        输出：
            变换用的矩阵
        """ 
        I = np.identity(A.shape[0])
        I[[ind,ind+np.argmax(A[ind:,ind])],:] = I[[ind+np.argmax(A[ind:,ind]),ind],:]  # 使用单位阵交换行作为置换矩阵
        return I
    
    def G_i(self,A,ind):
        """这部分计算第三类初等变换的矩阵Gi和Gi的逆矩阵Gi'
        参数：
            A(np array)： 输入的变换的矩阵
            ind(int): 选择行
        输出：
            变换用的矩阵
        """
        I = np.identity(A.shape[0])
        a_i = A[ind:,ind]
        a_i = a_i/np.max(a_i)
        a_i[0],a_i[np.argmax(a_i)] =  a_i[np.argmax(a_i)],a_i[0]
        I[ind:,ind] = a_i
        I1 = I.copy()
        I1[ind+1:,ind] = (-1)*I1[ind+1:,ind]  # 注意这个才是真正的第三类初等变换矩阵，下面返回的是初等变换矩阵及其逆矩阵
        return I1,I 
    
    #类似我们手工的操作方法
    def PLU_trans(self):
        A = self.A
        n = A.shape[0]
        L = np.identity(n)
        # 定义这两个list用于存放变换的矩阵
        Pi = [];Gi = [];Gi_inv = [] 
        # 这里按照操作的顺序进行处理
        for i in range(n-1):
            P_i = self.P_i(A,i)
            Pi.append(P_i)
            A = np.dot(P_i,A)
            L[:,:i] = np.dot(P_i,L[:,:i])  # 注意实际上L矩阵前面的部分是随着行变换发生变换的
            G_i = self.G_i(A,i)
            Gi.append(G_i[0])
            Gi_inv.append(G_i[1])
            A = np.dot(G_i[0],A)
            L[:,i] = G_i[1][:,i]  # 注意这里是添加的Gi'逆矩阵
        self.Pi = Pi;self.Gi = Gi;self.Gi_inv = Gi_inv 
        U = A
        Pi = Pi[::-1]  # Pi需要倒序下，方便乘积
        P = np.linalg.multi_dot(Pi)
        print("****manipulation PLU****")
        print("U","L","P");print(U,"\n",L,"\n",P)
        print("LU","PA");print(np.dot(L,U),"\n",np.dot(P,self.A))
        return L,U,P
    
    # 计算方程组Ax=b求解
    def sol_calc(self,L,U,P,b1):
        """计算方程组的求解
        参数:
            L,U,P(array): 分解出来的PLU三个矩阵
            b1(array): 想要求解的b
        
        会显示求解的值x，并且会验证
        """
        A = self.A
        n = A.shape[0]
        b = np.dot(P,b1)  # Ax=b  PAx=Pb  LUx=Pb  PA=Lu
        yi = []
        xi = []
        # 先求解下三角L对应的方程组
        for j in range(n):
            if j==0:
                yi.append(b[j])
            else:
                yi.append(b[j]-np.dot(L[j,:j],(np.array(yi)).T))
        # 再求解上三角U对应的方程组
        for k in range(n):
            if k==0:
                xi.append(yi[n-1-k]/U[n-1-k,n-1-k])
            else:    
                xi.append((yi[n-1-k]-np.dot(U[n-1-k,n-1-k+1:],(np.array(xi[::-1])).T))/U[n-1-k,n-1-k])
        xi =np.array(xi[::-1])
        
        np.set_printoptions(suppress=True)
        print("the nonsingular system")
        print(A,b1)
        print("the solutions:")
        print(xi)
        print("Ax")
        print(np.dot(A,xi))
    
    def det_calc(self,A,L,U,P):
        """计算行列式
        参数:
            A：原来的矩阵
            L,U,P(array): 分解出来的PLU三个矩阵
        
        会显示求解的行列式值，同时会显示用来验证的使用numpy计算的行列式
        """
        detU = 1
        inv_num = []
        inverse = 0
        n,_ = self.A.shape
        # 计算U的行列式，对角线乘积
        for i in range(n):
            detU = detU*U[i,i]
        # 计算逆序数
        for i in range(n):
            inv_num.append(np.argmax(P[i]))
        # 计算逆序数，这里考虑输入规模不大，就不用归并法了
        for i in range(1, n):
            for j in range(0, i):
                if inv_num[j] > inv_num[i]:
                    inverse += 1
        detA = detU/(-1)**inverse
        
        print("det(A)")
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
    
    plu = PLU(A)
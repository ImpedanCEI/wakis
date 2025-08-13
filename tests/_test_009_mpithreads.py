import os

# Set before importing numpy/scipy/mkl modules
os.environ["OMP_NUM_THREADS"] = "10"               # Number of OpenMP threads
os.environ["KMP_AFFINITY"] = "compact,granularity=fine"


from mpi4py import MPI
import numpy as np
from scipy.sparse import csc_matrix as sparse_mat, diags, vstack, hstack
try:
    from sparse_dot_mkl import csr_matrix, dot_product_mkl
except: pass
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info 
import timeit

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

nt = 1//p #int(os.environ['SLURM_CPUS_PER_TASK']) # num threads per task
user_api = threadpool_info()[0]['user_api']
#threadctl = threadpool_limits(limits=nt, user_api='blas')
#threadctl = threadpool_limits(limits=nt, user_api='openmp')
N = 30000000//p # Matrix size 

# Build curl
Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
Py = diags([-1, 1], [0, 10], shape=(N, N), dtype=np.int8)
Pz = diags([-1, 1], [0, 100], shape=(N, N), dtype=np.int8)
A = vstack([
                    hstack([sparse_mat((N,N)), -Pz, Py]),
                    hstack([Pz, sparse_mat((N,N)), -Px]),
                    hstack([-Py, Px, sparse_mat((N,N))])
                ], dtype=np.float64)

B = np.random.rand(3*N) 

try:
    A = csr_matrix(A)
except:
    A = A.tocsr()  

print(B.shape)

def test_matmul():
    #with threadpool_limits(limits=nt, user_api='blas'):
    #print(threadpool_info())
    #A = np.random.rand(N, N)
    #B = np.random.rand(1, N)  
    #C = np.matmul(A, B)
    
    try:
        C = dot_product_mkl(A,B)
    except:
        C = A.dot(B)
    print(C.shape)

#if my_rank == 0:
n_runs = 10
t = timeit.timeit("test_matmul()", 
                  setup="from __main__ import test_matmul", 
                  number=n_runs)

print(f"Rank {my_rank}: Average time over {n_runs} runs: {t/n_runs:.4f} seconds")


MPI.Finalize()

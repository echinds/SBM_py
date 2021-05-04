import cython
from cpython cimport array
from libc.math cimport exp
from libc.stdlib cimport srand, rand, RAND_MAX, malloc,free
from cython.view cimport array as cvarray
from cython.parallel import prange
cimport openmp

@cython.cdivision(True)
@cython.boundscheck(False)
def MC(double[:] w,n:int,l:int, Q:int, delta_T:int, seed:int):
    cdef int i,m,k,other_pos,pos,cur_aa,dq,new_aa
    cdef double dE,r
    cdef int L=l;
    cdef int q=Q;
    cdef int N=n;
    cdef int delta_t=delta_T;

    MSAr = cvarray(shape=(N*L,), itemsize=sizeof(int), format="i")
    cdef int [:] MSA = MSAr

    srand(seed)
    rand()
    for i in range(N*L):
        MSA[i] =  <int>(1.0*rand()/(RAND_MAX)*q)
    for m in prange(N,nogil=True,schedule='static'):          
        for k in range(delta_t):
            pos= <int>(1.0*rand()/(RAND_MAX)*L);
            cur_aa=MSA[m*L+pos]
            dq=1 + <int>(1.0*rand()/(RAND_MAX)*(q-1));
            
            new_aa = (cur_aa + dq) % q 
            
            dE = w[<int>(L * (L - 1) / 2 * q * q + (pos ) * q + new_aa )] - \
                w[<int>(L * (L - 1) / 2 * q * q + (pos ) * q + cur_aa )];

            for other_pos in range(pos+1,L):
                dE = dE+ w[<int>(q * q * ((pos ) * (L * 2 - pos-1) / 2 + (other_pos - pos - 1)) + \
                    (new_aa ) * q + MSA[m*L+other_pos ] )] - \
                    w[<int>(q * q * ((pos ) * (L * 2 - pos-1) / 2 + (other_pos - pos - 1)) + \
                    (cur_aa ) * q + MSA[m*L+other_pos ] )];


            for other_pos in range(pos): 
                 dE = dE+ w[<int>(q * q *((other_pos ) * (L * 2 - other_pos-1) / 2 + \
                     (pos - other_pos - 1)) + (MSA[m*L+other_pos ] ) * q + new_aa )] - \
                     w[<int>(q * q * ((other_pos ) * (L * 2 - other_pos-1) / 2 +  \
                     (pos - other_pos - 1)) + (MSA[m*L+other_pos ] ) * q + cur_aa )];
    
            r=(1.0*rand()/(RAND_MAX))/exp(dE)
            
            if r< 1 :
                MSA[m*L+pos]=new_aa;

    return MSA #array.array('i',[x for x in MSA[0:N*L]])
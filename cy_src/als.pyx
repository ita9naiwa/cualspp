import os
from tqdm import tqdm

cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange

import numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset
cimport scipy.linalg.cython_blas as cython_blas
cimport scipy.linalg.cython_lapack as cython_lapack

cdef extern from "../src/als.h":
    void _cuALS(int *dev_indices, int *dev_indptr, float *dev_data, int nnz,
                float *X, float *Y, int n_users, int n_items, int d, float reg)

    void load_matrix_to_cuda_memory(int *indices, int *indptr, float *data,
                                    int nnz, int n_users,
                                    int **dev_indices, int **dev_indptr, float** dev_data)

    void load_factors_to_cuda_memory(float *X, float *Y,
                                     int n_users, int n_items, int d,
                                     float **dev_X, float **dev_Y)

    void finalize(float *X, float *Y, float *dev_X, float *dev_Y,
                  int *u_indices, int *u_indptr, float *u_data,
                  int *i_indices, int *i_indptr, float *i_data,
                  int n_users, int n_items, int d)

def get_mat_info(mat):
    return mat.indices, mat.indptr, mat.data

def cuALS(user_item_matrix, d, reg, max_iter):
    n_users, n_items = user_item_matrix.shape
    nnz = user_item_matrix.nnz
    item_user_matrix = user_item_matrix.transpose().tocsr()
    _u_indices, _u_indptr, _u_data = get_mat_info(user_item_matrix)
    cdef int[:] u_indices = _u_indices
    cdef int[:] u_indptr = _u_indptr
    cdef float[:] u_data = _u_data

    _i_indices, _i_indptr, _i_data = get_mat_info(item_user_matrix)

    cdef int[:] i_indices = _i_indices
    cdef int[:] i_indptr = _i_indptr
    cdef float[:] i_data = _i_data

    cdef float[:, :] X = np.random.normal(0, 0.01, size=(n_users, d)).astype(np.float32)
    cdef float[:, :] Y = np.random.normal(0, 0.01, size=(n_items, d)).astype(np.float32)
    cdef int *dev_u_indices = NULL
    cdef int *dev_u_indptr = NULL
    cdef float *dev_u_data = NULL
    cdef int *dev_i_indices = NULL
    cdef int *dev_i_indptr = NULL
    cdef float *dev_i_data = NULL
    cdef float *dev_X = NULL
    cdef float *dev_Y = NULL
    load_matrix_to_cuda_memory(&u_indices[0], &u_indptr[0], &u_data[0],
                               nnz, n_users,
                               &dev_u_indices, &dev_u_indptr, &dev_u_data)

    load_matrix_to_cuda_memory(&i_indices[0], &i_indptr[0], &i_data[0],
                               nnz, n_items,
                               &dev_i_indices, &dev_i_indptr, &dev_i_data)

    load_factors_to_cuda_memory(&X[0, 0], &Y[0, 0],
                                n_users, n_items, d,
                                &dev_X, &dev_Y)

    for i in tqdm(range(max_iter)):
        _cuALS(dev_u_indices, dev_u_indptr, dev_u_data, nnz, dev_X, dev_Y,
                    n_users, n_items, d, reg)
        _cuALS(dev_i_indices, dev_i_indptr, dev_i_data, nnz, dev_Y, dev_X,
                    n_items, n_users, d, reg)
    finalize(&X[0, 0], &Y[0, 0], dev_X, dev_Y,
    dev_u_indices, dev_u_indptr, dev_u_data, dev_i_indices, dev_i_indptr, dev_i_data,
    n_users, n_items, d)
    return np.asarray(X), np.asarray(Y)

def cyALS(user_item_matrix, d, reg, max_iter, method='cpu_cg', num_threads=0, with_loss=False):
    if num_threads == 0:
        num_threads = os.cpu_count()
    n_users, n_items = user_item_matrix.shape
    item_user_matrix = user_item_matrix.transpose().tocsr()
    u_indices, u_indptr, u_data = get_mat_info(user_item_matrix)
    i_indices, i_indptr, i_data = get_mat_info(item_user_matrix)
    X = np.random.normal(0, 0.01, size=(n_users, d)).astype(np.float32)
    Y = np.random.normal(0, 0.01, size=(n_items, d)).astype(np.float32)
    for ep in tqdm(range(max_iter)):
        if method == 'cg':
            # user iter
            _cyALS_iter_CG(u_indices, u_indptr, u_data, X, Y, n_users, n_items, d, reg, num_threads)
            # item iter
            _cyALS_iter_CG(i_indices, i_indptr, i_data, Y, X, n_items, n_users, d, reg, num_threads)
        elif method == 'naive':
            # user iter
            _cyALS_iter(u_indices, u_indptr, u_data, X, Y, n_users, n_items, d, reg,num_threads)
            # item iter
            _cyALS_iter(i_indices, i_indptr, i_data, Y, X, n_items, n_users, d, reg, num_threads)
        else:
            _cyALS_iter(u_indices, u_indptr, u_data, X, Y, n_users, n_items, d, reg,num_threads)
            # item iter
            _cyALS_iter(i_indices, i_indptr, i_data, Y, X, n_items, n_users, d, reg, num_threads)
        if with_loss:
            loss = _calculate_loss(u_indptr, u_indices, u_data, X, Y, reg)
            print("loss at", ep, loss)
    return X, Y

@cython.cdivision(True)
@cython.boundscheck(False)
def _cyALS_iter(int[:] indices, int[:]indptr, float[:] data,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d, float reg, int num_threads):
    cdef float[:, :] YTY = np.dot(Y.T, Y)
    cdef float[:, :] base_A = YTY + reg * np.eye(d, dtype=np.float32)
    cdef int one = 1
    cdef float zero = 0.0
    cdef float c_ui = 0.0
    cdef float temp = 0.0
    cdef int i, j, idx, u
    cdef int err
    cdef float *A
    cdef float *b
    with nogil, parallel(num_threads=num_threads):
        A = <float*> malloc(d * d * sizeof(float));
        b = <float*> malloc(d * sizeof(float))
        for u in prange(n_users, schedule='dynamic', chunksize=8):
            memcpy(A, &base_A[0, 0], d * d * sizeof(float))
            memset(b, 0, d * sizeof(float))
            for idx in range(indptr[u], indptr[u + 1]):
                c_ui = data[idx]
                i = indices[idx]
                # B
                cython_blas.saxpy(&d, &c_ui, &Y[i, 0], &one, b, &one)
                for j in range(d):
                    temp = (c_ui - 1.0) * Y[i, j]
                    cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, A + d * j, &one)
            err = 0
            cython_lapack.sposv(b"U", &d, &one, A, &d, b, &d, &err)

            if err == 0:
                memcpy(&X[u, 0], b, d * sizeof(float))
            else:
                with gil:
                    raise ValueError("cython_lapack.posv failed (err=%i) on row %i. Try "
                                    "increasing the regularization parameter." % (err, u))
        free(A)
        free(b)


@cython.cdivision(True)
@cython.boundscheck(False)
def _cyALS_iter_CG(int[:] indices, int[:]indptr, float[:] data,
           float[:, :] X, float[:, :] Y,
           int n_users, int n_items, int d, float reg, int num_threads):
    cdef float[:, :] YTY = np.dot(Y.T, Y)
    cdef float[:, :] base_A = YTY + reg * np.eye(d, dtype=np.float32)
    cdef int one = 1
    cdef float zero = 0.0
    cdef float c_ui = 0.0
    cdef float temp = 0.0
    cdef float mone = -1.0, fone = 1.0
    cdef float alpha, beta
    cdef int i, j, idx, u
    cdef float *r
    cdef float *p
    cdef float *Ap
    cdef float pAp, rtr, rtr_new
    cdef int cg_i
    #print(base_A[0, 0], base_A[0, 1], base_A[0, 2], base_A[0, 3])
    with nogil, parallel(num_threads=1):
        r = <float*> malloc(d * sizeof(float))
        p = <float*> malloc(d * sizeof(float))
        Ap = <float*> malloc(d * sizeof(float))
        for u in prange(n_users, schedule='dynamic', chunksize=8):
            if indptr[u] == indptr[u + 1]:
                continue
            # calculating r_0
            # https://github.com/benfred/implicit/blob/main/implicit/cpu/_als.pyx#L188
            cython_blas.ssymv(b'U', &d, &mone, &base_A[0, 0], &d, &X[u, 0], &one, &zero, r, &one)
            for idx in range(indptr[u], indptr[u + 1]):
                c_ui = data[idx]
                i = indices[idx]
                # B
                temp = cython_blas.sdot(&d, &Y[i, 0], &one, &X[u, 0], &one)
                temp = c_ui - (c_ui - 1.0) * temp
                cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, r, &one)

            # https://en.wikipedia.org/wiki/Conjugate_gradient_method
            memcpy(p, r, d * sizeof(float))

            rtr = cython_blas.sdot(&d, r, &one, r, &one)
            if rtr <= 1e-10:
                break
            for cg_i in range(1):
                # calculating Ap, without actually calculating (YTCuY)p
                # it decomposes (YTY)p + (YT(Cu - 1Y)p
                cython_blas.ssymv(b'U', &d, &fone, &base_A[0, 0], &d, p, &one, &zero, Ap, &one)
                for idx in range(indptr[u], indptr[u + 1]):
                    i = indices[idx]
                    c_ui = data[idx]

                    temp = (c_ui - 1.0) * cython_blas.sdot(&d, &Y[i, 0], &one, p, &one)
                    cython_blas.saxpy(&d, &temp, &Y[i, 0], &one, Ap, &one)
                # calculating pAp
                pAp = cython_blas.sdot(&d, p, &one, Ap, &one)

                alpha = rtr / pAp
                cython_blas.saxpy(&d, &alpha, p, &one, &X[u, 0], &one)
                alpha = -alpha
                cython_blas.saxpy(&d, &alpha, Ap, &one, r, &one)
                rtr_new = cython_blas.sdot(&d, r, &one, r, &one)
                if rtr_new <= 1e-10:
                    break
                beta = rtr_new / rtr
                for j in range(d):
                    p[j] = r[j] + beta * p[j]
                rtr = rtr_new
            #with gil:
            #    print(X[u, 0], X[u, 1], X[u, 2], X[u, 3])
        free(r)
        free(p)
        free(Ap)





@cython.cdivision(True)
@cython.boundscheck(False)
def _calculate_loss(int[:] indptr, int[:] indices, float[:] data,
                    float[:, :] X, float[:, :] Y, float reg,
                    int num_threads=0):

    cdef int users = X.shape[0], items = Y.shape[0]
    cdef int u, i, idx
    cdef int d = X.shape[1]
    cdef int one = 1
    cdef float c_ui, temp
    cdef float fzero = 0, fone =1, fmone = -1
    cdef float[:, :] YtY = np.dot(np.transpose(Y), Y)
    cdef float *left
    cdef float *x
    cdef float *y
    cdef double loss = 0, total_confidence = 0, item_norm = 0, user_norm = 0
    with nogil, parallel(num_threads=num_threads):
        left = <float *> malloc(sizeof(float) * d)
        for u in prange(users, schedule='dynamic', chunksize=8):
            x = &X[u, 0]
            cython_blas.ssymv(b"U", &d, &fone, &YtY[0, 0], &d, x, &one, &fzero, left, &one)
            for idx in range(indptr[u], indptr[u + 1]):
                i = indices[idx]
                c_ui = data[idx]
                y = &Y[i, 0]
                temp = -2 * c_ui + (c_ui - 1) * cython_blas.sdot(&d, y, &one, x, &one)
                cython_blas.saxpy(&d, &temp, y, &one, left, &one)
                loss += c_ui
            loss += cython_blas.sdot(&d, left, &one, x, &one)
            user_norm += cython_blas.sdot(&d, x, &one, x, &one)
        for i in prange(items, schedule='dynamic', chunksize=8):
            y = &Y[i, 0]
            item_norm += cython_blas.sdot(&d, y, &one, y, &one)
        free(left)
    loss += reg * (item_norm + user_norm)
    return loss

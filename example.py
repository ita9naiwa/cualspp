import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from cuals import cyALS, cuALS
import time
import numpy as np
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares as ALS


_, ratings = get_movielens('1m')
ratings = ratings.T.tocsr()
tr, te = train_test_split(ratings, 0.8)


def model_eval(X, Y, tr, te, K=10):
    """
        this exploits implicit's evaluation features
    """
    model = ALS(use_gpu=False)
    model.user_factors = X
    model.item_factors = Y
    return ranking_metrics_at_k(model, tr, te, K=K)


dim = 1024
reg = 1.0
max_iter = 10

implicit = ALS(factors=dim, iterations=max_iter, use_gpu=True, use_cg=True, num_threads=0)
prev = time.time()
implicit.fit(tr)
print("implicit ALS with CUDA: runtime", time.time() - prev)

implicit = ALS(factors=dim, iterations=max_iter, use_gpu=False, use_cg=True, num_threads=0)
prev = time.time()
implicit.fit(tr)
print("implicit ALS without CUDA: runtime", time.time() - prev)

prev = time.time()
X, Y = cyALS(tr, d=dim, reg=reg, max_iter=max_iter, num_threads=0, method='cg')
print("cyALS: runtime", time.time() - prev)

prev = time.time()
X, Y = cuALS(tr, d=dim, reg=reg, max_iter=max_iter)
print("cuALS runtime", time.time() - prev)
print(model_eval(X, Y, tr, te))

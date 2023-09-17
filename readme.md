## ials++ with CUDA

This is an implemenation of [ialspp](https://arxiv.org/abs/2110.14044) algorithm, which allows ALS training in very high dimension (>1024) with CUDA (cuALS)

This also contains basic implementation of ALS with conjugate gradient (cyALS)

#### Example
```python
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

X, Y = cuALS(tr, d=dim, reg=reg, max_iter=max_iter)

def model_eval(X, Y, tr, te, K=10):
    """
        this exploits implicit's evaluation features
    """
    model = ALS(use_gpu=False)
    model.user_factors = X
    model.item_factors = Y
    return ranking_metrics_at_k(model, tr, te, K=K)
print(model_eval(X, Y, tr, te))
```

# rcda/tasks/c_metrics.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double evaluate_mc(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred):
    cdef Py_ssize_t n = y_true.shape[0]
    cdef double mean_true = 0.0, mean_pred = 0.0
    cdef double cov = 0.0, var_true = 0.0, var_pred = 0.0
    cdef Py_ssize_t i
    cdef double diff_true, diff_pred

    for i in range(n):
        mean_true += y_true[i]
        mean_pred += y_pred[i]

    mean_true /= n
    mean_pred /= n

    for i in range(n):
        diff_true = y_true[i] - mean_true
        diff_pred = y_pred[i] - mean_pred
        cov += diff_true * diff_pred
        var_true += diff_true ** 2
        var_pred += diff_pred ** 2

    cov /= n - 1
    var_true /= n - 1
    var_pred /= n - 1

    if var_true == 0 or var_pred == 0:
        return np.nan
    else:
        return (cov ** 2) / (var_true * var_pred)

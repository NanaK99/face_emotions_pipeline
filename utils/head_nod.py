import scipy
import numpy as np


def f_test(arr1, arr2):
    f, p = scipy.stats.f_oneway(arr1, arr2)
    return f, p


def detect_head_nod(head_nod_len, normal_headnod_mean, normal_headnod_std, headnod_lefts, headnod_rights):
    normal = np.random.normal(normal_headnod_mean, normal_headnod_std, head_nod_len)

    lefts_p = f_test(headnod_lefts, normal)
    rights_p = f_test(headnod_rights, normal)
    if rights_p[1] or lefts_p[1] <= 0.1:
        return "HEAD NODE"
    else:
        return ""
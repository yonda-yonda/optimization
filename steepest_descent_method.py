#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import types
import numbers
import math

class DoNotConvergeException (Exception):
  def __str__ (self):
    return ("Don't converge.")

class FailedArmijoRuleException (Exception):
  def __str__ (self):
    return ("Failed armijo rule.")

def steepest_descent_method(objective_func, gradient_func, x0, max_iteration=None, eps=1e-6, alpha = 0.001, beta = 0.5, max_power = 32):
    '''
    最急降下法
    '''
    assert isinstance(x0, np.matrixlib.defmatrix.matrix) and x0.shape[1] == 1, \
        'x0はnumpyのmatrixかつ1次元配列である必要があります。'
    x_dim = x0.shape[0]
    x_mat = x0.copy()

    assert isinstance(objective_func, types.FunctionType), \
        'objective_funcは関数である必要があります。'
    f = objective_func(x_mat)
    assert not isinstance(f, bool) and isinstance(f, numbers.Number), \
        'objective_funcの返り値は数値(スカラー)である必要があります。'


    assert isinstance(gradient_func, types.FunctionType), \
        'gradient_funcは関数である必要があります。'
    df_mat = gradient_func(x_mat)
    assert df_mat.shape[0] == x_dim and df_mat.shape[1] == 1, \
        'gradient_funcの返り値はnumpyのmatrixかつ長さx_dimの1次元配列である必要があります。'

    if max_iteration is None:
        max_iteration = max(100, 20 * x_dim)

    iteration = 0
    while iteration < max_iteration:
        iteration = iteration + 1

        if np.linalg.norm(df_mat) < eps:
            break

        # 最急降下法
        d_mat = -1.0 * df_mat

        # Armijoルールによるステップ探索
        armijo_power = 0
        while armijo_power < max_power:
            t = math.pow(beta, armijo_power)
            if objective_func(x_mat + t*d_mat) <= f + alpha * t * df_mat.T * d_mat:
                break
            armijo_power = armijo_power + 1

        if armijo_power >= max_power:
            raise FailedArmijoRuleException()

        x_mat = x_mat + t * d_mat
        df_mat = gradient_func(x_mat)
        f = objective_func(x_mat)

    if iteration >= max_iteration:
        raise DoNotConvergeException()

    return x_mat


if __name__ == '__main__':
    # 目的関数
    def objective(x):
        return 2 * x[0,0] - 4 * x[1,0] + x[0,0] ** 2 + 2 * x[1,0] ** 2 + 2 * x[0,0] * x[1,0]

    def gradient(x):
        return np.mat([[2 + 2*x[0,0] + 2*x[1,0]],[-4 + 4 * x[1,0] + 2*x[0,0]]])

    x0 = np.mat([[1],[1]])

    print(steepest_descent_method(objective, gradient, x0))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import types
import numbers
import math
import sys
import os
import io

path = os.path.join(os.path.dirname(__file__), './')
sys.path.append(path)
from golden_section_method import *


class ConjugateGradientMethod(object):
    '''
        共役勾配法
    '''
    def __init__(self, objective_func, x_dim, gradient_func=None, max_iteration=None, eps=None):
        self.x_dim = x_dim
        x = np.mat(np.random.randn(self.x_dim,1))
        assert isinstance(objective_func, types.FunctionType), \
            'objective_funcは関数である必要があります。'

        f = objective_func(x)
        assert not isinstance(f, bool) and isinstance(f, numbers.Number), \
            'objective_funcの返り値は数値(スカラー)である必要があります。'

        self.objective_func = objective_func

        if gradient_func is None:
            self.gradient_func = self.calculate_gradient
        else:
            assert isinstance(gradient_func, types.FunctionType), \
                'gradient_funcは関数である必要があります。'
            df_mat = gradient_func(x)
            assert df_mat.shape[0] == self.x_dim and df_mat.shape[1] == 1, \
                'gradient_funcの返り値はnumpyのmatrixかつ長さx_dimの1次元配列である必要があります。'
            self.gradient_func = gradient_func


        # 終了条件
        if eps is not None:
            self.eps = eps
        else:
            self.eps = 1e-6
        if max_iteration is None:
            self.max_iteration = max(100, 20 * x_dim)
        else:
            self.max_iteration = max_iteration

        # 微分計算
        self.diff = 0.001

        # モード
        self.line_search_success = False
        self.converge = False

        # アルミホのルールによる直線探索パラメータ
        self.armijo_alpha = 0.001
        self.armijo_beta = 0.5
        self.armijo_max_iteration = 64

        # 直線探索
        self.step_size = 1e-4

        # 目的変数
        self.X = np.mat(np.empty((self.x_dim,self.max_iteration)), dtype=np.float64)
        self.__k = 0

    def calculate_gradient(self, x):
        shift_mat = (x * np.ones(self.x_dim) + self.diff * np.mat(np.identity(self.x_dim))).T
        return np.mat([(self.objective_func(shift_x.T) - self.objective_func(x))/self.diff for shift_x in shift_mat]).T

    def solve(self, x0, armijo_mode = True):
        assert isinstance(x0, np.matrixlib.defmatrix.matrix) and x0.shape[0] == self.x_dim and x0.shape[1] == 1, \
            'x0はnumpyのmatrixかつx_dimの1次元配列である必要があります。'

        # 目的変数初期化
        self.X[:, 0] = x0.copy()
        # フラグ初期化
        self.line_search_success = False
        self.converge = False


        hessian = np.mat(np.identity(self.x_dim))
        self.__k = 0

        while self.__k < self.max_iteration - 1:
            x_mat = self.X[:, self.__k]
            f = self.objective_func(x_mat)
            df_mat = self.gradient_func(x_mat)

            if np.linalg.norm(df_mat) < self.eps:
                break

            # 共役勾配法
            if self.__k > 0:
                gamma = ((df_mat.T * df_mat) / (df_pre_mat.T * df_pre_mat))[0,0]
                d_mat = -1.0 * df_mat + gamma * d_mat
            else:
                d_mat = -1.0 * df_mat


            if armijo_mode:
                # Armijoルールによるステップ探索
                armijo_power = 0
                while armijo_power < self.armijo_max_iteration:
                    t = math.pow(self.armijo_beta , armijo_power)
                    if self.objective_func(x_mat + t*d_mat) <= f + self.armijo_alpha * t * df_mat.T * d_mat:
                        break
                    armijo_power = armijo_power + 1

                if armijo_power >= self.armijo_max_iteration:
                    return None
                else:
                    self.line_search_success = True

            else:
                def line_objective_function(t):
                    return self.objective_func(x_mat + t * d_mat)

                t = golden_section_method(line_objective_function, 0.0, self.step_size)


            df_pre_mat = df_mat
            self.__k = self.__k + 1
            self.X[:, self.__k] = x_mat + t * d_mat

        if not self.__k >= self.max_iteration - 1:
            self.converge = True

        return self.X[:, self.__k]

    def get_objective_values(self):
        return self.X[:,:self.__k+1]

    def get_iteration(self):
        return self.__k + 1


if __name__ == '__main__':
    print('ConjugateGradientMethod')

    print('problem1')
    def objective1(x):
        return 2 * x[0,0] - 4 * x[1,0] + x[0,0] ** 2 + 2 * x[1,0] ** 2 + 2 * x[0,0] * x[1,0]

    def gradient1(x):
        return np.mat([[2 + 2*x[0,0] + 2*x[1,0]],[-4 + 4 * x[1,0] + 2*x[0,0]]])

    x1 = np.mat([[1],[10]])

    c=ConjugateGradientMethod(objective1, 2, gradient1)
    print('armijo', c.solve(x1).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x1, armijo_mode=False).T, c.converge, c.get_iteration())

    print('problem2')
    def objective2(x):
        return x[0,0] ** 2 + 2 * x[1,0] ** 2 - 1.0 * x[0,0] * x[1,0] + x[0,0] - 2.0 * x[1,0]

    def gradient2(x):
        return np.mat([[2*x[0,0] -1.0*x[1,0] + 1.0],[4 * x[1,0] -1.0*x[0,0] -2.0]])

    x2 = np.mat([[15],[15]])

    c=ConjugateGradientMethod(objective2, 2, gradient2)
    print('armijo', c.solve(x2).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x2, armijo_mode=False).T, c.converge, c.get_iteration())

    print('problem3')
    def objective3(x):
        return x[0,0] ** 2 + 2 * x[1,0] ** 2 - 1.0 * x[0,0] * x[1,0] + x[0,0] - 2.0 * x[1,0] \
            + 4.0 * math.sin(0.1 * (x[0,0] + 0.2857)**2) + 12.0 * math.sin(0.1 * (x[1,0] - 0.4286)**2)

    def gradient3(x):
        return np.mat([\
            [2*x[0,0] -1.0*x[1,0] + 1.0 + 0.8 * (x[0,0] + 0.2857) * math.cos(0.1 * (x[0,0] + 0.2857)**2)],\
            [4 * x[1,0] -1.0*x[0,0] -2.0 + 2.4 * (x[1,0] - 0.4286) * math.cos(0.1 * (x[1,0] - 0.4286)**2)]\
            ])

    x3 = np.mat([[15],[15]])

    c=ConjugateGradientMethod(objective3, 2, gradient3)
    print('armijo', c.solve(x3).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x3, armijo_mode=False).T, c.converge, c.get_iteration())

    print('problem4')
    def objective4(x):
        return 100.0*(x[1,0] - x[0,0]**2)**2 + (1-x[0,0])**2

    def gradient4(x):
        return np.mat([[-400.0*x[0,0]*(x[1,0]-x[0,0]**2) - 2*(1-x[0,0])],[200.0*(x[1,0]-x[0,0]**2)]])

    x4 = np.mat([[0],[0]])

    c=ConjugateGradientMethod(objective4, 2, gradient4)
    print('armijo', c.solve(x4).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x4, armijo_mode=False).T, c.converge, c.get_iteration())

    print('problem5')
    def objective5(x):
        return (1.5 - x[0,0]*(1 - x[1,0]))**2 + (2.25 - x[0,0] * (1-x[1,0]**2))**2 + (2.625 - x[0,0] * (1-x[1,0]**3))**2

    def gradient5(x):
        return np.mat([\
            [ -2.0*(1 - x[1,0])*(1.5 - x[0,0]*(1 - x[1,0])) -2.0 * (1-x[1,0]**2) * (2.25 - x[0,0] * (1-x[1,0]**2)) -2.0 * (1-x[1,0]**3) *(2.625 - x[0,0] * (1-x[1,0]**3))],\
            [ 2.0*x[0,0]*(1.5 - x[0,0]*(1 - x[1,0])) +4.0 *x[0,0]*x[1,0]*(2.25 - x[0,0] * (1-x[1,0]**2)) + 6.0*x[0,0]*x[1,0]*x[1,0]*(2.625 - x[0,0] * (1-x[1,0]**3))]\
            ])

    x5 = np.mat([[0],[0]])
    c=ConjugateGradientMethod(objective5, 2, gradient5)
    print('armijo', c.solve(x5).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x5, armijo_mode=False).T, c.converge, c.get_iteration())

    c=ConjugateGradientMethod(objective5, 2)
    print('勾配を解析的に与えない場合(' + str(c.diff) + ')')
    print('armijo', c.solve(x5).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x5, armijo_mode=False).T, c.converge, c.get_iteration())
    c.diff = 1e-8
    print('xの差を小さくする(' + str(c.diff) + ')')
    print('armijo', c.solve(x5).T, c.converge, c.get_iteration())
    print('optim ', c.solve(x5, armijo_mode=False).T, c.converge, c.get_iteration())
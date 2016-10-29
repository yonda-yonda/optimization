#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import numbers

class FailureQuadraticAproximationException (Exception):
  def __str__ (self):
    return ("Failed quadratic aproximation.")

class DoNotConvergeException (Exception):
  def __str__ (self):
    return ("Don't converge.")

def line_search(func, x0, step_size = 0.1, max_itration = 10000, eps = 1e-10):
    """
        二次多項式近似による直線探索
        一次元探索(最小化問題)
    """
    assert isinstance(func, types.FunctionType), \
        'funcは関数である必要があります。'
    assert not isinstance(x0, bool) and isinstance(x0, numbers.Number), \
        'x0は数値(スカラー)である必要があります。'
    f0 = func(x0)
    assert not isinstance(f0, bool) and isinstance(f0, numbers.Number), \
        'fの返り値は数値(スカラー)である必要があります。'

    # ステップ情報
    first_step = step_size
    moveup = False

    # 終了判定
    iteration = 0
    converged = False

    # 初期値
    x = [0] * 4
    f = [0] * 4
    step = first_step
    # 起点
    x[1] = x0
    f[1] = func(x0)


    while iteration < max_itration and not converged:
        iteration = iteration + 1

        try:
            x[3] = x[1] + step
            f[3] = func(x[3])
        except OverflowError as e:
            raise DoNotConvergeException()

        if f[3] <= f[1]:
            step = step * 2
            x[0] = x[1]
            f[0] = f[1]
            x[1] = x[3]
            f[1] = f[3]

            moveup = True
        else:
            is_convex = False
            if moveup:
                x[2] = x[3] -0.5*step
                f[2] = func(x[2])

                min_index = f.index(min(f))

                if min_index>=2:
                    x[0:3] = x[1:4]
                    f[0:3] = f[1:4]

                # i=0~2の3点は下に凸
                is_convex = True
            else:
                x[0] = x[1] - first_step
                f[0] = func(x[0])

                if f[1]<f[0]:
                    x[2] = x[3]
                    f[2] = f[3]
                    # i=0~2の3点は下に凸
                    is_convex = True
                else:
                    # 初期ステップ
                    x[1] = x[0]
                    f[1] = f[0]
                    # 探索方向が単調増加となるため探索方向を変える
                    first_step = -1.0*first_step
                    step = 2.0 * first_step

                    moveup = True

            if is_convex:
                if abs(x[0] - x[2]) < eps:
                    # 収束
                    converged = True

                # f[0] > f[1], f[1] < f[2] となる3点で二次多項式近似した式の頂点を見つける
                try:
                    delta_x = 0.5 * step * (f[2] - f[0])/(f[0] - 2*f[1] + f[2])
                except ZeroDivisionError as e:
                    raise FailureQuadraticAproximationException()

                else:
                    # 見つけた頂点とx[1]のうち小さいほうを起点に再探索
                    x_provisional = x[1] - delta_x
                    f_provisional = func(x_provisional)
                    if f_provisional < f[1]:
                        x[1] = x_provisional
                        f[1] = f_provisional

                    first_step = 0.5 * step
                    step = first_step
                    moveup = False


    return x[1]


if __name__ == '__main__':
    # 目的関数
    def objective1(x):
        return x ** 4 + 3.0 * x ** 3 + 2.0 * x ** 2 + 1.0

    x1 = -10.0

    print('objective : x ** 4 + 3.0 * x ** 3 + 2.0 * x ** 2 + 1.0')
    print('x0 = ' + str(x1))
    print(line_search(objective1, x1))

    x2 = -2.0
    print('objective : x ** 3 - x ** 2 - 2.0 * x')
    print('x0 = ' + str(x2))
    try:
        print(line_search(objective1, x2))
    except FailureQuadraticAproximationException as e:
        print(e)


    def objective2(x):
        return x ** 3 - x ** 2 - 2.0 * x

    x3 = 0.0

    print('objective : x ** 3 - x ** 2 - 2.0 * x')
    print('x0 = ' + str(x3))
    print(line_search(objective2, x3))

    x4 = -10.0

    print('objective : x ** 3 - x ** 2 - 2.0 * x')
    print('x0 = ' + str(x4))
    try:
        print(line_search(objective2, x4))
    except DoNotConvergeException as e:
        print(e)

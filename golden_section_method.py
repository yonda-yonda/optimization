#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import numbers

class CanNotFindConvexException (Exception):
  def __str__ (self):
    return ("Can't find convex.")

class DoNotConvergeException (Exception):
  def __str__ (self):
    return ("Don't converge.")


def golden_section_method(f, df, x0, step_size = 0.1, max_itration = 10000, eps = 1e-10):
    """
        黄金分割法
        一次元探索(最小化問題)
    """
    assert isinstance(f, types.FunctionType), \
        'fは関数である必要があります。'
    assert isinstance(df, types.FunctionType), \
        'dfは関数である必要があります。'
    assert not isinstance(x0, bool) and isinstance(x0, numbers.Number), \
        'x0は数値(スカラー)である必要があります。'

    f0 = f(x0)
    assert not isinstance(f0, bool) and isinstance(f0, numbers.Number), \
        'fの返り値は数値(スカラー)である必要があります。'
    df0 = df(x0)
    assert not isinstance(df0, bool) and isinstance(df0, numbers.Number), \
        'fの返り値は数値(スカラー)である必要があります。'

    # ステップ情報
    step = step_size
    direction = -1.0 * df0 # 目的関数が減少する方向へ
    x1 = x0
    f1 = f0

    iteration = 0
    # 頂点の反対側のxを見つける
    while iteration < max_itration:
        iteration = iteration + 1

        x2 = x1 + step * direction
        f2 = f(x2)

        if f1 < f2:
            break

        step = 2.0 * step

    if iteration >= max_itration:
        raise CanNotFindConvexException()

    iteration = 0
    # 黄金分割法で頂点に近づいていく
    tau = 1.618033988749895 # 黄金比 (1 + √5)/2
    while iteration < max_itration:
        iteration = iteration + 1

        width = abs(x1 - x2)
        if width < eps:
            break

        x3 = x1 + (tau - 1)/tau * width
        x4 = x1 + width / tau
        f3 = f(x3)
        f4 = f(x4)
        if f3 > f4:
            x1 = x3
        else:
            x2 = x4

    if iteration >= max_itration:
        raise DoNotConvergeException()

    return x1



if __name__ == '__main__':
    # 目的関数
    def objective(x):
        return x * x * x * x + 3.0 * x * x * x + 2.0 * x * x + 1.0

    # 極地は-1.640388203(下に凸), -0.609611797(上に凸), 0.0(下に凸)
    def gradient(x):
        return 4.0  * x * x * x + 9.0 * x * x + 4.0 * x

    x = -2.0

    print(golden_section_method(objective, gradient, x))
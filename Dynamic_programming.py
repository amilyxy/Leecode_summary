# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Dynamic_programming
   Description :
   Author :         amilyxy
   date：           2019/9/9
-------------------------------------------------
"""
'''
70. Climbing Stairs: 爬楼梯
describe: 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
          每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
'''
class Solution:
    # 方法一： 暴力穷举法（输入到35的时候就超出时间限制 ==）
    def climbStairs(self, n: int) -> int:
        def climb_stairs(i, n):
            if i > n:
                return 0
            if i == n:
                return 1
            return climb_stairs(i+1, n)+climb_stairs(i+2, n)
        res = climb_stairs(0, n)
        return res

    # 方法二：动态规划/斐波那契数
    def climbStairs(self, n: int) -> int:
        stage1 = 1
        stage2 = 2
        res = 0
        if n == 1:
            return stage1
        if n == 2:
            return stage2
        for j in range(3, n+1):
            res = stage1 + stage2
            stage1 = stage2
            stage2 = res
        return res
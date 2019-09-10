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

'''
62. Unique Paths: 不同路径
describe: 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
          机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
          问总共有多少条不同的路径？
'''
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 超出时间限制
        if m == 0 or n == 0:
            return 1
        def findpath(i, j):
            if i == (m-1) or j == (n-1):
                return 1
            return (findpath(i+1, j)+ findpath(i, j+1))
        res = findpath(0, 0)
        return res

    # 方法一 排列组合
    def uniquePaths(self, m: int, n: int) -> int:
        def jiec(i, j):
            res = 1
            for k in range(j, j-i, -1):
                res = k*res
            return res
        steps = jiec(n-1, m+n-2)//jiec(n-1, n-1)
        return steps

    # 方法二 题解的动态规划法
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
    # 官方提供的还有两种优化算法，可以看一下，有点难理解










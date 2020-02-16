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

    # 新增排列组合方法 C(总的步数, 走二阶的步数)
    def climbStairs(self, n: int) -> int:
        stage2 = n//2
        rest = 0
        def jiec(a, b):
            res = 1
            for k in range(a, a-b,-1):
                res = k*res
            return res
        for i in range(0, stage2+1):
            rest += jiec((n-i), i)//jiec(i, i)
            # print(rest)
        return rest

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
    # 方法三 优化一
    def uniquePaths(self, m: int, n: int) -> int:
        pre = [1] * n
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] = pre[j] + cur[j-1]
            pre = cur[:]
        return pre[-1]
    # 方法三 优化二
    def uniquePaths(self, m: int, n: int) -> int:
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] += cur[j - 1]
        return cur[-1]

'''
63. Unique PathsII: 不同路径II
describe: 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
          机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
          现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
'''
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        # 方法一
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        pre = [1] * n
        col = [1] * m
        cur = [1] * n
        for k in range(n):
            if obstacleGrid[0][k] == 1:
                pre[k:] = [0] * (n - k)
                break
        # print(pre)
        for z in range(m):
            if obstacleGrid[z][0] == 1:
                col[z:] = [0] * (m - z)
                break
        # print(col)
        if m == 1:
            return pre[-1]
        if n == 1:
            return col[-1]
        for i in range(1, m):
            for j in range(1, n):
                # 对每一列第一个 初始化
                cur[0] = col[i]
                if obstacleGrid[i][j] == 1:
                    cur[j] = 0
                    continue
                else:
                    cur[j] = pre[j] + cur[j - 1]
            pre = cur[:]
        return cur[-1]

    # 题解方法
    # 亮点： 给dp加一个边界，就类似于62题的方法三 优化二
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        m, n = len(obstacleGrid[0]), len(obstacleGrid)
        dp = [1] + [0] * m
        for i in range(0, n):
            for j in range(0, m):
                dp[j] = 0 if obstacleGrid[i][j] else dp[j] + dp[j - 1]
        return dp[-2]

'''
120 Triangle 三角形最小路径和
'''
# 按照惯例 第一个依旧是我的超时方法
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 三角形的深度
        n = len(triangle)
        res = []
        def helper(deep, i, tmp):
            if deep == n:
                res.append(sum(tmp))
            else:
                helper(deep+1, i, tmp+[triangle[deep][i]])
                helper(deep+1, i+1,tmp+[triangle[deep][i+1]])
        helper(1, 0, [triangle[0][0]])
        return min(res)

# 当时有想过自底向上 但仔细想了 @名字太长了不想打
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        for i in range(len(triangle)-2, -1, -1):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
        return triangle[0][0]

# 自顶向上也是可以做的 @eternalhunter
class Solution(object):
    def minimumTotal(self, triangle):
        if not triangle:
            return 0
        if len(triangle) == 1:
            return triangle[0][0]
        for i in range(1, len(triangle)):
            triangle[i][0] += triangle[i-1][0]
            for j in range(1, len(triangle[i])-1):
                triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
            triangle[i][-1] += triangle[i-1][-1]
        return min(triangle[-1])

'''
279 完全平方和
'''
# 来看看我妖娆的代码... 其实这个方法在比较大的数值就容易超出时间限制了 不妥不妥
import math
class Solution:
    def numSquares(self, n: int) -> int:
        # 先用一种笨方法
        maxpow = math.floor(math.sqrt(n))
        li = set(map(lambda x: pow(x, 2), range(1,maxpow+1)))
        newli = set(li)
        if n in li:
            return 1
        k = 1
        while 1:
            k+=1
            temp = set()
            for i in newli:
                if i<n:
                    for j in li:
                        if i+j == n:
                            return k
                        temp.add(i+j)
            newli = set(temp)

# 好好看看人家的方法 思路和你的一毛一样 时间差的不是一点半点... @powcai
class Solution:
    def numSquares(self, n: int) -> int:
        from collections import deque
        if n == 0 or n == 1: return n
        if int(n ** 0.5) ** 2 == n: return 1
        queue = deque([n])
        candidates = set([i ** 2 for i in range(1, int(n ** 0.5) + 1)])
        step = 0
        while queue:
            step += 1
            l = len(queue)
            for _ in range(l):
                tmp = queue.pop()
                for x in candidates:
                    val = tmp - x
                    if val in candidates:
                        return step + 1
                    elif val > 0:
                        queue.appendleft(val)

# 根据第二个代码修改了一下第一个代码 速度确实快了很多啊
class Solution:
    def numSquares(self, n: int) -> int:
        # 先用一种笨方法
        maxpow = math.floor(math.sqrt(n))
        cand = set(map(lambda x: x**2, range(1,maxpow+1)))
        if n in cand:
            return 1
        k = 0
        newli = {n}
        while 1:
            k+=1
            temp = set()
            for i in cand:
                for j in newli:
                    a = j-i
                    if a in cand:
                        return k+1
                    temp.add(a)
            newli = temp

# 拉格朗日四个方形定理 @QQqun902025048
class Solution:
    def numSquares(self, n: int) -> int:
        while n % 4 == 0:
            n /= 4
        if n % 8 == 7:
            return 4
        a = 0
        while a ** 2 <= n:
            b = int((n - a ** 2) ** 0.5)
            if a ** 2 + b ** 2 == n:
                return bool(a) + bool(b)
            a += 1
        return 3

# 动态规划方法 @QQqun902025048
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0]
        for i in range(1, n+1):
            dp.append(min(dp[-j*j] for j in range(1, 1 + int(i**0.5))) + 1)
        return dp[-1]











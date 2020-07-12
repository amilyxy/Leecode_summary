# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Matrix
   Description :
   Author :         amilyxy
   date：           2019/9/23
-------------------------------------------------
"""
'''
63. Rotate Image: 旋转图像
describe: 给定一个 n × n 的二维矩阵表示一个图像，将图像顺时针旋转90度
          必须原地旋转矩阵
'''
import math
class Solution:
    # 我的思路： 从外而内对矩每个值进行旋转
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix[0])-1
        ring = math.ceil(len(matrix[0])/2)
        for i in range(ring):
            # 列
            col = i
            for j in range(i, n-i):
                # 行
                row = j
                temp =  matrix[row][col]
                for z in range(4):
                    # 下一个值给temp
                    row1 = col
                    col1 = (n - row)
                    temp1 = matrix[row1][col1]
                    matrix[row1][col1] = temp
                    col = col1
                    row = row1
                    temp = temp1
                    # 上面可以改成（多利用多元赋值！！
                    # row1, col1 = col, (n - row)
                    # matrix[row1][col1], temp = temp, matrix[row1][col1]
                    # row, col = row1, col1

    # 题解方法 转置加翻转（我怎么就没观察出这个规律呢！！
    def rotate(self, matrix):
        n = len(matrix[0])
        # transpose matrix
        for i in range(n):
            for j in range(i, n):
                # 多元赋值还是秀啊
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
                # reverse each row
        for i in range(n):
            matrix[i].reverse()

    # 题解方法 三
    # 感觉这个也挺好的 将四次循环用四个多元赋值表示 比较简洁
    def rotate(self, matrix):
        n = len(matrix[0])
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp

'''
54. Spiral Matrix: 螺旋矩阵
describe: 给定一个包含 m x n 个元素的矩阵（m 行, n 列）
          请按照顺时针螺旋顺序，返回矩阵中的所有元素。
'''
class Solution:
    # 题解方法就不写了
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        res = []
        if len(matrix) == 0:
            return res
        n = len(matrix[0])
        m = len(matrix)
        ring = (min(m, n) + 1) // 2
        for i in range(ring):
            # 行
            row, col = i, list(range(i, n - i))
            res.extend([matrix[row][z] for z in col])
            print(res)

            row, col = list(range(i + 1, m - i)), col[-1]
            res.extend([matrix[z][col] for z in row])

            if len(row) != 0:
                row, col = row[-1], list(range(n - i - 2, i - 1, -1))
                res.extend([matrix[row][z] for z in col])
            else:
                return res

            if len(col) != 0:
                row, col = list(range(m - i - 2, i, -1)), col[-1]
                res.extend([matrix[z][col] for z in row])
            else:
                return res

        return res
    # 题解简单方法（比较难理解
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = list(map(list, zip(*matrix)))[::-1]
            print(matrix)
        return res

# 最近做剑指offer发现的新的做法 标记数组
# 只是我觉得最一目了然的做法
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 标记做法
        res = []
        dire = [[0, 1], [1, 0], [0, -1], [-1, 0]]  #方向右下左上
        m = len(matrix)
        if m:
            n = len(matrix[0])
        else:
            return matrix
        marked = [[0 for _ in range(n)] for _ in range(m)]
        i, j, k= 0, 0, 0
        while 0<=i<m and 0<=j<n and not marked[i][j]:
            res.append(matrix[i][j])
            marked[i][j] = 1
            while 0<= i+dire[k][0] <m and 0<=j+dire[k][1]<n and not marked[i+dire[k][0]][j+dire[k][1]]:
                i+=dire[k][0]
                j+=dire[k][1]
                res.append(matrix[i][j])
                marked[i][j] = 1
            k = (k+1)%4
            i = i+dire[k][0]
            j = j+dire[k][1]
        return res

'''
59. Spiral Matrix: 螺旋矩阵II
describe: 给定一个正整数 n，生成一个包含 1 到 n2 所有元素
          且元素按顺时针顺序螺旋排列的正方形矩阵。
'''
class Solution:
    def generateMatrix(self, n: int) -> list[list[int]]:
        res = list(range(1, n * n + 1))
        matrix = [[0 for i in range(n)] for j in range(n)]
        ring = (n + 1) // 2
        for i in range(ring):
            # 行
            row, col = i, list(range(i, n - i))
            for z in col:
                matrix[row][z] = res[0]
                res.pop(0)

            row, col = list(range(i + 1, n - i)), col[-1]
            for z in row:
                matrix[z][col] = res[0]
                res.pop(0)

            if len(row) != 0:
                row, col = row[-1], list(range(n - i - 2, i - 1, -1))
                for z in col:
                    matrix[row][z] = res[0]
                    res.pop(0)
            else:
                return matrix

            if len(col) != 0:
                row, col = list(range(n - i - 2, i, -1)), col[-1]
                for z in row:
                    matrix[z][col] = res[0]
                    res.pop(0)
            else:
                return matrix

        return matrix
# 方法二
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        dire = [[0, 1], [1, 0], [0, -1], [-1, 0]]  #方向右下左上
        marked = [[0 for _ in range(n)] for _ in range(n)]
        i, j, k, count = 0, 0, 0, 1
        while 0<=i<n and 0<=j<n and not marked[i][j]:
            marked[i][j] = count
            count += 1
            while 0<= i+dire[k][0] <n and 0<=j+dire[k][1]<n and not marked[i+dire[k][0]][j+dire[k][1]]:
                i+=dire[k][0]
                j+=dire[k][1]
                marked[i][j] = count
                count += 1
            k = (k+1)%4
            i = i+dire[k][0]
            j = j+dire[k][1]
        return marked

'''
73. Set Matrix Zeroes 矩阵置零
'''
# m+n的来了 好吧 看了一圈好像我写的这个是最复杂的 需要额外的存储空间
class Solution:
    def setZeroes(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        setzeroi = set()
        setzeroj = set()
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    setzeroi.add(i)
                    setzeroj.add(j)
        for i in range(m):
            for j in range(n):
                if i in setzeroi or j in setzeroj:
                    matrix[i][j] = 0

# O(1)的作法  @powcai(好强
class Solution(object):
    def setZeroes(self, matrix):
        is_col = False
        R = len(matrix)
        C = len(matrix[0])
        for i in range(R):
            if matrix[i][0] == 0: is_col = True
            for j in range(1, C):
                if matrix[i][j]  == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0

        for i in range(R-1,-1,-1):
            for j in range(C-1,0,-1):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0
            if is_col: matrix[i][0] = 0


'''
329. Longest Increasing Path in a Matrix 矩阵中的最长递增路径
'''
# 写了好久终于写出来了 运行时间有点长 804ms
# 看到答案有先排序在搜索的方法  确实挺巧妙的 我的方法还可以照这个再改进
class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        m = len(matrix)
        if m == 0:
            return 0
        else:
            n = len(matrix[0])
        dire = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        maxlen = [[-1 for _ in range(n)] for _ in range(m)]  # 矩阵中每个值都是都是其最长递增路径

        def helper(i, j):
            temp = [0]
            for k in dire:
                newi, newj = i + k[0], j + k[1]
                if 0 <= newi < m and 0 <= newj < n and matrix[newi][newj] > matrix[i][j]:
                    if maxlen[newi][newj] != -1:
                        temp.append(1 + maxlen[newi][newj])
                    else:
                        temp.append(1 + helper(newi, newj))
            maxlen[i][j] = max(temp)
            return max(temp)

        for i in range(m):
            for j in range(n):
                if maxlen[i][j] == -1:
                    maxlen[i][j] = helper(i, j)

        print(maxlen)
        return max(max(row) for row in maxlen) + 1

# 评论里面与众不同的一个方法 加了排序步骤
class Solution(object):
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append((matrix[i][j], i, j))
        lst.sort()
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for num, i, j in lst:
            dp[i][j] = 1
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = i + di, j + dj
                if 0 <= r < m and 0 <= c < n:
                    if matrix[i][j] > matrix[r][c]:
                        dp[i][j] = max(dp[i][j], 1 + dp[r][c])
        return max([dp[i][j] for i in range(m) for j in range(n)])

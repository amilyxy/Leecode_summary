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

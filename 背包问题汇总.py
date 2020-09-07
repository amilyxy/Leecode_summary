# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       背包问题汇总
   Description :
   Author :         amilyxy
   date：           2020/3/20
-------------------------------------------------
"""
'''
补一下 01背包问题 牛客网题目
其中c为背包容量 n为物品总数
做法一： 时间复杂度O(cn), 空间复杂度O(cn)  牛客网上通过90% （搞不懂 // 顺便写一下物品序号
做法二： 优化空间复杂度为O(n)
'''
import sys
inp = []
while True:
    line = sys.stdin.readline().strip()
    if line == '':
        break
    item = line.split(' ')
    inp.append([int(i) for i in item])

v = inp[0][0]
n = inp[0][1]
nums = inp[1:]
# 升序
# nums.sort(key = lambda x:x[0])
res = [[0 for _ in range(v + 1)] for _ in range(n + 1)]

def bag():
    for i in range(1, n + 1):
        for j in range(1, v + 1):
            if j >= nums[i - 1][0]:
                res[i][j] = max(res[i - 1][j - nums[i - 1][0]] + nums[i - 1][1], res[i - 1][j])
            else:
                res[i][j] = res[i - 1][j]

    print(res[-1][-1])

# 找出具体放入背包的物品
def show():
    has = [0]*n
    j = c
    for i in range(n, 0, -1):
        if res[i][j]>res[i-1][j]:
            has.append(i)
        j -= nums[i][0]
    return has

def bag1():
    res = [0 for _ in range(v+1)]
    for i in range(1, n+1):
        for j in range(v, 0, -1):
            if j>=nums[i-1][0]:
                res[j] = max(res[j-nums[i-1][0]]+nums[i-1][1], res[j])

    print(res[-1])
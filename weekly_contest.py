# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       week_competition
   Description :
   Author :         amilyxy
   date：           2019/12/23
-------------------------------------------------
"""
'''
5292: 划分数组为连续数字集合
'''
# 老样子 第一个还是超时程序 在36个案例时超时
class Solution:
    def isPossibleDivide(self, nums: list[int], k: int) -> bool:
        nums.sort(reverse=True)
        while len(nums):
            count = 0
            last = nums[-1]-1
            n = len(nums)
            for i in range(n-1, -1, -1):
                print(nums[i])
                print(last)
                if count<=k-1:
                    if nums[i] != last:
                        if (nums[i]-last) != 1:
                            return False
                        last = nums[i]
                        count+=1
                        nums.pop(i)
                else:
                    break
            if count<=k-1:
                return False
        return True

# 哈哈哈哈做出来了！
from collections import Counter
class Solution:
    def isPossibleDivide(self, nums: list[int], k: int) -> bool:
        nums = Counter(nums)
        if len(nums) == k and len(set(nums.values())) == 1:
            return True
        else:
            while len(nums):
                cut = {}
                keys = set(nums)
                begin = min(keys)
                end = begin+k
                freq = nums[begin]
                for i in range(begin, end):
                    if i in keys and nums[i] >= freq:
                        cut[i] = freq
                    else:
                        return False
                nums = nums-Counter(cut)
            return True

'''
5296. 两棵二叉搜索树中的所有元素
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        self.res = []

        def helper(node):
            if node is None:
                return
            helper(node.left)
            self.res.append(node.val)
            helper(node.right)

        helper(root1)
        helper(root2)
        self.res.sort()
        return self.res

'''
5297. 跳跃游戏 III
'''
from collections import defaultdict


class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        # 先构建一个下标图
        dictarr = defaultdict(set)
        n = len(arr)
        for i in range(n):
            val = arr[i]
            if i - val >= 0:
                dictarr[i].add((i - val))
            if i + val < n:
                dictarr[i].add((i + val))
                # print(dictarr)

        def dfs(a, marked):
            if arr[a] == 0:
                return True
            for i in dictarr[a]:
                if i not in marked:
                    marked.append(i)
                    if dfs(i, marked): return True

        marked = [start]
        if dfs(start, marked):
            return True
        else:
            return False

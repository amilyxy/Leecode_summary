# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Backtracking
   Description :
   Author :         amilyxy
   date：           2019/9/6
-------------------------------------------------
"""
'''
78. Subsets: 子集
describe: 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
'''
import itertools
# 今天的好像都是评论方法  雨我无瓜
class Solution:
    #  方法一 库函数法
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = []
        for i in range(len(nums)+1):
            for tmp in itertools.combinations(nums, i):
                res.append(tmp)
        return res

    # 方法二 迭代
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
            # res.extend([[i] + z for z in res])
        return res

    # 方法三 递归（回溯）算法
    '''
    ⭐ 加精!
    '''
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = []
        n = len(nums)
        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, n):
                helper(j + 1, tmp + [nums[j]])
        helper(0, [])
        return res

    # 方法四：二进制掩码的方法
    def subsets(self, nums: List[int]) -> List[List[int]]:
        size = len(nums)
        n = 1 << size
        res = []
        for i in range(n):
            cur = []
            for j in range(size):
                if i >> j & 1:
                    cur.append(nums[j])
            res.append(cur)
        return res

    # 方法五：

'''
90. SubsetsII: 子集II
describe: 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）
          解集不能包含重复的子集。
'''
class Solution:
    # 方法一 先找出所有子集 在逐个判断是否重复
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        res = [[]]
        out = []
        for i in nums:
            res += [[i] + z for z in res]
        for j in res:
            if sorted(j) in out:
                continue
            else:
                out.append(sorted(j))
        return out

    # 以下题解方法
    # 方法二：
    



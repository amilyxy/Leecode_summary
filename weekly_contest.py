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
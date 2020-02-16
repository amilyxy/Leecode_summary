# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       sort_algorithm
   Description :
   Author :         amilyxy
   date：           2020/2/5
-------------------------------------------------
"""
'''
merge sort 归并排序
包括两种实现方法：自顶向下（递归实现）、自底向上（迭代实现）
'''
# 自顶向下 好像时间有点长
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums
        else:
            mid = (len(nums) + 1) // 2
            return self.merge(self.sortArray(nums[:mid]), self.sortArray(nums[mid:]))

    def merge(self, sub1, sub2):
        tmp = []
        while sub1 and sub2:
            if sub1[0] < sub2[0]:
                tmp.append(sub1.pop(0))
            else:
                tmp.append(sub2.pop(0))
        return tmp + (sub1 or sub2)

# 递归的简单版
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<2:
            return nums
        mid = (len(nums)+1)//2
        sub1 = self.sortArray(nums[:mid])
        sub2 = self.sortArray(nums[mid:])
        tmp = []
        while sub1 and sub2:
            if sub1[0]<sub2[0]:
                tmp.append(sub1.pop(0))
            else:
                tmp.append(sub2.pop(0))
        return tmp+(sub1 or sub2)


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Binary_Search
   Description :
   Author :         amilyxy
   date：           2019/9/21
-------------------------------------------------
"""
'''
278. First Bad Version: 第一个错误的版本
'''
def isBadVersion():
    pass
class Solution:
    # 自己写的迭代方法
    def firstBadVersion(self, n):
        li = range(1, n+1)
        l = n
        while l>1:
            inter = l//2
            if isBadVersion(li[inter]):
                li = li[:inter]
            else:
                li = li[inter:]
            l = len(li)
        if isBadVersion(li[0]):
            return li[0]
        else:
            return li[0]+1

    # 突然发现还是设置left和right比较好
    # 根据题解方法写的：
    def firstBadVersion(self, n):
        l, r = 1, n
        while l < r:
            # 这里的(r-l)//2还是很关键的  防止某些编程语言越界
            mid = l + (r - l) // 2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid + 1
        return l

    # 递归方法
    def firstBadVersion(self, n):
        def badhelper(l, r):
            if l == r:
                return l
            mid = l+(r-l)//2
            if isBadVersion(mid):
                l = badhelper(l, mid)
            else:
                l = badhelper(mid+1, r)
            return l
        return badhelper(1, n)

'''
35. Search Insert Position 搜索插入位置
describe: 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
          你可以假设数组中无重复元素。
'''
class Solution:
    # 迭代方法
    # 按理说也可以改成递归方法
    def searchInsert(self, nums: list[int], target: int) -> int:
        if len(nums) == 1:
            if target > nums[0]:
                return 1
            else:
                return 0
        l, r = 0, (len(nums) - 1)
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] > target:
                r = mid - 1
            if nums[mid] < target:
                l = mid + 1
            if nums[mid] == target:
                return mid
        if nums[l] >= target:
            return l
        else:
            return l + 1
    '''
    ⭐ 加精!  
    看完题解之后的简化版本
    '''
    def searchInsert(self, nums: list[int], target: int) -> int:
        l, r = 0, (len(nums) - 1)
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] > target:
                r = mid - 1
            if nums[mid] < target:
                l = mid + 1
            if nums[mid] == target:
                return mid
        return l

'''
33. Search in Rotated Sorted Array 搜索旋转排序数组
describe: 略略略
'''
class Solution:
    # 太难了呀 要搞明白+1 -1
    def search(self, nums: list[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        l, r = 0, (len(nums) - 1)
        while l <= r:
            mid = l + (r - l) // 2
            # 说明左边是顺的
            if nums[mid] == target:
                return mid
            if nums[l] <= nums[mid]:
                if nums[l] <= target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # 说明右边顺的
            else:
                if nums[mid] <= target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

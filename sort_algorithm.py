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
时间复杂度：O(nlog(2)n) 空间复杂度：O(n)
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

'''
Bubble Sort 冒泡排序（交换排序）
时间复杂度O(n^2) 空间复杂度O(1) 稳定
'''
class Solution:
    def BubbleSort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(0, n-i-1):
                if nums[j]>nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        return nums

'''
Quick sort 快速排序
简单选择排序的进阶版
时间复杂度O(nlogn) 空间复杂度O(logn) 不稳定
'''
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums)

        def helper(nums, left, right):
            if left >= right:
                return
            mid = self.quicksort(nums, left, right)
            helper(nums, left, mid - 1)
            helper(nums, mid + 1, right)

        helper(nums, left, right - 1)
        return nums

    def quicksort(self, nums, left, right):
        base = left
        while left < right:
            while left < right and nums[right] >= nums[base]:
                right -= 1
            while left < right and nums[left] <= nums[base]:
                left += 1
            nums[left], nums[right] = nums[right], nums[left]
        nums[left], nums[base] = nums[base], nums[left]
        return left

# 另一种写法
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1

        def helper(left, right):
            if left < right:
                mid = self.quicksort(nums, left, right)
                helper(left, mid - 1)
                helper(mid + 1, right)

        helper(left, right)
        return nums

    def quicksort(self, nums, left, right):
        base = left
        idx = left + 1
        i = idx
        while i <= right:
            if nums[i] < nums[base]:
                nums[i], nums[idx] = nums[idx], nums[i]
                idx += 1
            i += 1
        nums[idx - 1], nums[base] = nums[base], nums[idx - 1]
        return idx - 1

'''
Insert Sort 插入排序
时间复杂度O(n^2)  空间复杂度O(1) 稳定
'''
class Solution:
    def InsertSort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(1, n):
            j = i
            tmp = nums[i]
            while j>0 and nums[j-1]>tmp:
                nums[j] = nums[j-1]
                j -= 1
            nums[j] = tmp
        return nums

'''
Shell Sort 希尔排序
插入排序的进阶版
时间复杂度 O(nlogn)  空间复杂度O(1)  不稳定
'''
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                preidx = i - gap
                tmp = nums[i]
                while preidx >= 0 and nums[preidx] > tmp:
                    nums[preidx + gap] = nums[preidx]
                    preidx -= gap
                nums[preidx + gap] = tmp
            gap //= 2
        return nums




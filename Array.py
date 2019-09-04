# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Array
   Description :
   Author :         amilyxy
   date：           2019/9/4
-------------------------------------------------
"""
'''
27. Remove Element: 移除元素
describe: 给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
'''
class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        # 方法一
        nums[:] = list(filter(lambda x: x != val, nums))  # 加不加list其实都一样
        return len(nums)

        # 方法二
        L = len(nums)
        for i in range(L):
            try:
                nums.remove(val)
            except:
                break
        return len(nums)

        # 方法三
        j=len(nums)
        for i in range(j-1,-1,-1):
            if nums[i]==val:
                nums.pop(i)
        return len(nums)

'''
26. Remove Duplicates from Sorted Array: 删除排序数组中的重复项
describe: 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
'''
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # 方法一：时间太长 2400ms+
        a = list(set(nums))
        a.sort(key = nums.index)
        j = 0
        for i in a:
            nums[j] = i
            j += 1
        return j

        # 方法二：
        a = set(nums)
        j = 0
        for i in range(len(nums)):
            if nums[i] in a:
                nums[j] = nums[i]
                a.remove(nums[i])
                j += 1
                # 判断a是否为空
                if a.isEmpty():
                    break
        return j

        # # 这个很好啊 为啥会超出时间限制
        # j = 0
        # for i in range(len(nums)):
        #     if nums[i] not in nums[:i]:
        #         nums[j] = nums[i]
        #         j += 1
        # return j

        j = 1
        if len(nums) == 0:
            return 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[j] = nums[i]
                j += 1
        return j

'''
80. Remove Duplicates from Sorted Array II: 删除排序数组中的重复项II
describe: 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
'''
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # 方法一 80ms左右
        j = 1
        count = 0
        if len(nums) == 0: return 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                if count == 0:
                    count = 1
                    nums[j] = nums[i]
                    j += 1
            else:
                nums[j] = nums[i]
                count = 0
                j += 1
        return j

        # 方法二：时间好像不太行 感觉pop操作比赋值操作时间要长
        for i in range(len(nums)-1, 1, -1):
            if nums[i] == nums[i-2]:
                nums.pop(i)
        return len(nums)

        # 方法三：这个是本来有思路 但是以为不可行 看了下面那个又写了
        j = 2
        if len(nums) == 0: return 0
        if len(nums) == 1: return 1
        for i in range(2, len(nums)):
            if nums[i] != nums[j-2]:  # 不能i-2！！！ j-2 很关键！
                nums[j] = nums[i]
                j += 1
        return j

        # 评论解法
        i = 0
        for e in nums:
            if i < 2 or e != nums[i - 2]:
                nums[i] = e
                i += 1
        return i




# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       bit
   Description :
   Author :         amilyxy
   date：           2019/10/5
-------------------------------------------------
"""
'''
389. Find the Difference: 找不同
describe: 给定两个字符串 s 和 t，它们只包含小写字母。
          字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
          请找出在 t 中被添加的字母。
'''
import operator
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        # 方法一：逐个删除
        # emm这个不太好，i in list时间复杂度高也就算了，还把s/t中的元素修改了
        t = list(t)
        for i in s:
            t.remove(i)
        res = "".join(t)
        return res
        # replace做不需要转换成list
        # for i in s:
        #     t = t.replace(i, '', 1)

    # 方法二 按位比较，就是str不能直接排序有点烦
    def findTheDifference(self, s: str, t: str) -> str:
        s = list(s)
        s.sort()
        s = "".join(s)
        t = list(t)
        t.sort()
        t = "".join(t)
        i = 0
        while i<len(s):
            if operator.ne(s[i], t[i]):
                return t[i]
            i += 1
        if i == len(s):
            return t[i]

    # 方法三：ASCII之差
    def findTheDifference(self, s: str, t: str) -> str:
        res = chr(sum(map(ord, t)), sum(map(ord, s)))
        return res

    # 方法四： 异或法
    '''
    ⭐ 加精!  
    感觉这个才是题目所要求的的呀！
    '''
    def findTheDifference(self, s: str, t: str) -> str:
        res = 0
        for i in s:
            res ^= ord(i)
        for j in t:
            res ^= ord(j)
        return chr(res)

'''
136. Single Number: 找不同
describe: 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
'''
class Solution:
    # 这个时间复杂度为n^2 不好不好
    def singleNumber(self, nums: list[int]) -> int:
        nums.sort()
        for i in range(0, len(nums), 2):
            # nums[i] != nums[i+1]
            if i == (len(nums)-1) or operator.ne(nums[i], nums[i+1]):
                return nums[i]

    # 方法二：位操作
    '''
    ⭐ 加精!  
    感觉这个才是题目所要求的的呀！
    '''
    class Solution(object):
        def singleNumber(self, nums):
            res = 0
            for i in nums:
                res ^= i
            # reduce(lambda x, y: x^y, nums)
            return res



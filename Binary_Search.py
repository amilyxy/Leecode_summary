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
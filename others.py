# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       others
   Description :
   Author :         amilyxy
   date：           2020/2/10
-------------------------------------------------
"""
'''
45.跳跃游戏II
其实本来应该放在贪心算法的章节，但是没有啊，那就放这里吧
'''
# 贪心（我写的
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            nums[i] = nums[i]+i
        res, count, i = 0, 0, i
        while res<n-1:
            tmp = res
            for j in range(res, min(n, nums[res]+1)):
                if j >=n-1:
                    return count+1
                if nums[j]>tmp:
                    tmp = nums[j]
                    res = j
            count+=1
        return count

# 贪心（评论方法
class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = 0
        end = 0
        maxposition = 0
        # 这里的len(nums)-1很精髓
        for i in range(len(nums) - 1):
            maxposition = max(maxposition, nums[i] + i)
            if i == end:
                end = maxposition
                steps += 1
        return steps
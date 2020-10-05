# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       一些力扣没有的面试题
   Description :
   Author :         amilyxy
   date：           2020/10/1
-------------------------------------------------
"""
'''
1.平方后不重复元素
desc：给定一个有序数组，返回平方之后不重复的元素
nums = [-5,-3,-1,-1,0,1,1,1,2]
'''
# 双指针方法一：遍历取绝对值最大，记录上一个数
def solution1(nums):
    i, j = 0, len(nums)-1
    res = []
    pre = None
    while i<=j:
        if abs(nums[i])>abs(nums[j]):
            if pre!=abs(nums[i]):
                res.append(pow(nums[i],2))
                pre = abs(nums[i])
            i+=1
        else:
            if pre!=abs(nums[j]):
                res.append(pow(nums[j],2))
                pre = abs(nums[j])
            j-=1
    return res

'''
2.数组中不重复元素
desc: 给定一个先递增后递减数组(双调数组)，返回不重复元素
case: [1,1,1,1], [-3,-3,-2,-1,-1,4,4,1,1,0,-1,-1,-2,-3,-3]
'''
def solution2(nums):
    i, j = 0, len(nums)-1
    res = []
    pre = None
    while i<=j:
        if nums[i]<nums[j]:
            if pre!=nums[i]:
                res.append(nums[i])
                pre = nums[i]
            i+=1
        else:
            if pre!=nums[j]:
                res.append(nums[j])
                pre = nums[j]
            j-=1
    return res

'''
3.面试题02.05 链表求和
desc：比较烦的是要用链表返回
'''
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(0)
        node = head
        tmp = 0
        while l1 or l2:
            if l1:
                tmp+=l1.val
            if l2:
                tmp+=l2.val
            node.next = ListNode(tmp % 10)
            tmp = tmp // 10
            node = node.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if tmp:
            node.next = ListNode(tmp)
        return head.next




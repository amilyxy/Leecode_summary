# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       LinkedList
   Description :
   Author :         amilyxy
   date：           2019/9/15
-------------------------------------------------
"""
'''
63. Reverse Linked List: 反转链表
describe: 反转一个单链表。
            示例:
            输入: 1->2->3->4->5->NULL
            输出: 5->4->3->2->1->NULL
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 题解-迭代方法
    def reverseList(self, head: listNode) -> listNode:
        pre = None
        cur = head
        while cur != None:
            nexttemp = cur.next
            cur.next = pre
            pre = cur
            cur = nexttemp
        return pre

    # 递归方法

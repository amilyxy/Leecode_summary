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
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    # 题解-迭代方法
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while cur != None:
            nexttemp = cur.next
            cur.next = pre
            pre = cur
            cur = nexttemp
        return pre

    # 递归方法
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        #获取下一个节点：
        next_node = head.next
        #递归反转
        res = self.reverseList(next_node)
        #将头节点接到反转链表的尾部
        next_node.next = head
        head.next = None
        return res

'''
141. Linked List Cycle: 环形链表
describe: 给定一个链表，判断链表中是否有环。
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    # 题解方法一 感觉不太好 修改了链表元素
    def hasCycle(self, head):
        if not head:
            return False
        while head.next and head.val != None:
            head.val = None  # 遍历的过程中将值置空
            head = head.next
        if not head.next:  # 如果碰到空发现已经结束，则无环
            return False
        return True  # 否则有环

    # 题解方法二 类似于官方题解里面的哈希表 没必要用dict
    def hasCycle(self, head):
        p = head
        st = set()
        while p:
            if p in st:
                return True
            st.add(p)
            p = p.next
        return False

    # 用python实现快慢指针？
    def hasCycle(self, head):
        if head == None or head.next == None:
            return False
        fast, slow = head.next, head
        while slow != fast:
            if fast == None or fast.next == None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True

'''
24. Swap Nodes in Pairs: 两两交换链表中的节点
describe: 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
          你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
'''
class Solution:
    # 自己写的-算是迭代法吧... 其实是有点繁琐的
    def swapPairs(self, head: ListNode) -> ListNode:
        # 初始化
        if head == None or head.next == None:
            return head
        else:
            pre = None
            cur = head
            nextnode = head.next
        # 交换
        while 1:
            temp = nextnode.next
            cur.next = temp
            nextnode.next = cur
            if pre == None:
                head = nextnode
                pass
            else:
                pre.next = nextnode
            if temp == None or temp.next == None:
                break
            pre = cur
            cur = temp
            nextnode = cur.next

    '''
    ⭐ 加精!  
    递归大法好！ 时候总结一下递归的套路了 移步总结
    '''
    def swapPairs(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        nextnode = head.next
        head.next = self.swapPairs(nextnode.next)
        nextnode.next = head
        return nextnode

    '''
    ⭐ 加精!  
    这里面的非递归版本二解法思想还挺不错的
    需要的节点比较少
    https://leetcode-cn.com/problems/swap-nodes-in-pairs/solution/di-gui-2chong-fei-di-gui-by-heng-29/
    '''



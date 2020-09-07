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

'''
328.Odd Even Linked List 奇偶链表
'''
# 根据数学归纳法写了好久...尝试失败... 以下是题解复现python版
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        evenhead=even=head.next
        odd = head
        while even and even.next:  # while odd.next and even.next:
            odd.next, even.next = odd.next.next, even.next.next
            odd, even = odd.next, even.next
        odd.next = evenhead
        return head

'''
92. Reverse Linked List II 反转链表
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 感觉是写的有点复杂嗷
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        count = 0
        begin, beginpre = head, head
        while begin:
            count+=1
            if count == m:
                end = count
                pre = begin
                cur = pre.next
                while end<n:
                    temp = cur.next
                    cur.next = pre
                    pre = cur
                    cur = temp
                    end+=1
                if count == 1:
                    head = pre
                else:
                    beginpre.next = pre
                begin.next = cur
                break
            else:
                beginpre = begin
                begin =begin.next
        return head

'''
21.合并两个有序链表  
'''
# 方法一 递归
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val > l2.val: l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1 or l2

# 方法二 迭代
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        preHead = ListNode(-1)
        prev = preHead
        while l1 and l2:
            if l1.val<l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2
        return preHead.next

'''
23.合并k个排序链表
'''
# 方法一 根据21题分而治之 5683ms 超级耗时
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if lists:
            prenode = lists[0]
        else:
            return

        def helper(node1, node2):
            prenode = ListNode(-1)
            prev = prenode
            while node1 and node2:
                if node1.val < node2.val:
                    prev.next = node1
                    node1 = node1.next
                else:
                    prev.next = node2
                    node2 = node2.next
                prev = prev.next
            prev.next = node1 or node2
            return prenode.next

        for i in range(1, len(lists)):
            prenode = helper(prenode, lists[i])

        return prenode

# 方法二 利用headq 题解方法@powcai
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next

# 官方题解有另一种暴力解法，list存储所有的val，然后重新构建链表，不是原地操作
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next

'''
138.复制带随机指针的链表
'''
# dfs方法 因为需要保存节点所以比较耗空间O(n)
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        lookup={}
        def dfs(node):
            if not node: return None
            if node in lookup: return lookup[node]
            copy = Node(node.val)
            lookup[node] = copy
            copy.next, copy.random = dfs(node.next), dfs(node.random)
            return copy
        return dfs(head)

# 方法二 线性时间方法
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None
        node = head
        while node:
            tmp = node.next
            node.next = Node(node.val)
            node.next.next = tmp
            node = tmp
        node = head
        while node:
            if node.random:
                node.next.random = node.random.next
            node = node.next.next
        # 拆分
        copy_head = head.next
        copy_pre = head
        copy_post = head.next

        while copy_pre:
            # pre
            copy_pre.next = copy_pre.next.next
            copy_pre = copy_pre.next
            # post
            if copy_pre:
                copy_post.next = copy_pre.next
                copy_post = copy_post.next
        return copy_head
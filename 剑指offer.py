# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       剑指offer
   Description :
   Author :         amilyxy
   date：           2020/1/23
-------------------------------------------------
"""
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

'''
1.数组中重复的数字
'''
# 方法一
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while i!=nums[i]:
                if nums[nums[i]] == nums[i]:
                    return nums[i]
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]

# 方法二：
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        setnum = set()
        for i in nums:
            if i not in setnum:
                setnum.add(i)
            else:
                return i

'''
2.二维数组中的查找
'''
# 原始方案：暴力查找 时间复杂度O(n^2)
class Solution:
    def Find(self, target, array):
        m = len(array)
        n = len(array[0])
        for i in range(m):
            for j in range(n):
                if array[i][j] == target:
                    return True
        return False

# 优化方案二 右上查询 时间复杂度O(m+n)
class Solution:
    def Find(self, target, array):
        if not array: return False
        n = len(array[0])
        row, col = 0, n-1
        while row<len(array) and col>-1:
            if array[row][col] == target:
                return True
            if array[row][col]<target:
                row+=1
            else:
                col-=1
        return False

'''
3. 替换空格
'''
class Solution:
    def replaceSpace(self, s):
        s = list(s)
        for i in range(len(s)-1, -1, -1):
            if s[i] == ' ':
                s[i] = '%20'
        return ''.join(s)

'''
4. 从尾到头打印链表
'''
# 方案一 顺序遍历逆序输出
class Solution:
    def printListFromTailToHead(self, listNode):
        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        return res[::-1]

# 方案二 回溯
class Solution:
    def printListFromTailToHead(self, listNode):
        # write code
        res = []
        def back(node):
            if node.next:
                back(node.next)
            res.append(node.val)
        if listNode:
            back(listNode)
        return res

'''
5. 重建二叉树
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, preorder, inorder):
        if not preorder:
            return None
        loc = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.reConstructBinaryTree(preorder[1: loc+1], inorder[0:loc])
        root.right = self.reConstructBinaryTree(preorder[loc+1:], inorder[loc+1:])
        return root

'''
6.二叉树的下一个节点
'''
class Solution:
    def GetNext(self, pNode):
        # write code here
        # 如果该节点有右子树
        if pNode.right:
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            return pNode
        else:
            if pNode.next:
                if pNode == pNode.next.left:
                    return pNode.next
                else:
                    while pNode.next.left != pNode:
                        pNode = pNode.next
                        if not pNode.next:
                            return None
                    return pNode.next
            else:
                return None

'''
7.用两个栈实现队列
'''
from collections import deque
# 栈先进后出 队列先进先出
class Solution:
    def __init__(self):
        self.stack1 = deque()
        self.stack2 = deque()

    def push(self, node):
        # write code here
        self.stack1.append(node)

    def pop(self):
        # return xx
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        if self.stack2:
            return self.stack2.pop()
        return -1

'''
8.菲波那切数列
牛客网上跳台阶、变态跳台阶、覆盖格子都是斐波那契的应用
'''
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n == 0: return 0
        if n == 1: return 1
        n1, n2 = 0, 1
        for i in range(2, n + 1):
            n1, n2 = n2, n1 + n2
        return n2

'''
9.旋转排序数组中的最小值
'''
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        l, r = 0, len(rotateArray)-1
        while l<r:
            mid = l+(r-l)//2
            if rotateArray[r]<rotateArray[mid]:
                l=mid+1
            else:
                r=mid
        return rotateArray[l]

# 带重复数字
class Solution:
    def minArray(self, numbers):
        l, r = 0, len(numbers)-1
        while l<r:
            if numbers[l] == numbers[r]:
                l += 1
            else:
                mid = l+(r-l)//2
                if numbers[mid]<=numbers[r]:
                    r = mid
                else:
                    l = mid+1
        return numbers[l]

'''
10.矩阵中的路径
力扣中相似题目有8、90、77、39
'''
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word: return True
        if not board: return False
        m, n = len(board), len(board[0])
        dire = [[-1, 0], [0, -1], [1, 0], [0, 1]]

        def helper(i, j, word, marked):
            if not word: return True
            marked[i][j] = 1
            for k in dire:
                new_i, new_j = i + k[0], j + k[1]
                if 0 <= new_i < m and 0 <= new_j < n and board[new_i][new_j] == word[0] and not marked[new_i][new_j]:
                    if helper(new_i, new_j, word[1:], marked): return True
            marked[i][j] = 0

        marked = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    if helper(i, j, word[1:], marked):
                        return True
        return False

'''
11. 机器人的运动范围
'''
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        dire = [[-1, 0], [0, 1], [1, 0], [0, 1]]
        marked = [[0 for _ in range(n)] for _ in range(m)]

        def bitsum(a):
            sum = 0
            while a > 0:
                sum += a % 10
                a = a // 10
            return sum

        def helper(row, col):
            marked[row][col] = 1
            for i in dire:
                newi, newj = row + i[0], col + i[1]
                if 0 <= newi < m and 0 <= newj < n and marked[newi][newj] == 0:
                    if bitsum(newi) + bitsum(newj) <= k:
                        helper(newi, newj)

        if k >= 0: helper(0, 0)
        return sum([sum(i) for i in marked])

'''
12.剪绳子
当n=10000时，测量一下题解一报错超过循环次数，题解二105s
'''
'''
解题一：递归方法（会存在许多重复步骤）31ms
'''
class Solution:
    def cutRope(self, number):
        # write code here
        def helper(num):
            res = []
            if num == 1:
                return 1
            for i in range(1, (num+1)//2+1):
                res.append(max(helper(i), i)*max(helper(num-i), num-i))
            return max(res)
        return helper(number)

'''
解题二：动态规划方法（避免重复） 28ms
'''
class Solution:
    def cutRope(self, number):
        # write code here
        res = [0, 1]
        # write code here
        for i in range(2, number+1):
            tmp = []
            for j in range(1, (i+1)//2+1):
                tmp.append(max(res[j],j) * max(res[i - j], (i-j)))
            res.append(max(tmp))
        return res[-1]

'''
13.二进制中1的个数
'''
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        if n < 0:
            n = n & 0xffffffff
        while n:
            count += 1
            # 补码中最右边有几个1，就循环几次
            n = (n - 1) & n
        return count

'''
14.数值的整数次方
'''
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0 and exponent<0:
            return "the input is invalid"
        else:
            return pow(base, exponent)  # base**exponent

# 不使用库函数
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0: return 0
        res = 1
        if n < 0: x, n = 1 / x, -n
        while n:
            if n & 1: res *= x
            x *= x
            n >>= 1
        return res

'''
15.打印从1到最大的n位数
'''
# 一般做法
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        res = []
        for i in range(1, pow(10, n)):
            res.append(i)
        return res

# 大数问题 用排列方法解决
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        base = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        res = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        tmp = res
        for i in range(1, n):
            tmp1 = []
            for k in tmp:
                for j in base:
                    tmp1.append(k+j)
            res.extend(tmp1)
            tmp = tmp1
        return [int(i) for i in res]

'''
16.删除链表的节点
'''
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val: return head.next
        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next
        if cur:
            pre.next = cur.next
        return head

'''
17.调整数组顺序使得奇数位于偶数前面
① 容易想到的方法
② 原地算法
'''
# 方法一
class Solution:
    def reOrderArray(self, array):
        # write code here
        a, b = [],[]
        for i in array:
            a.append(i) if i&0x01 else b.append(i)
        return a+b

# 方法二 算是原地算法吧？？？
class Solution:
    def reOrderArray(self, array):
        # write code here
        n = len(array)
        pre, cur = 0, 0
        while cur<n:
            if array[cur]&0x01:
                tmp = array[cur]
                for i in range(cur, pre-1, -1):
                    array[i] = array[i-1]
                array[pre] = tmp
                pre+=1
            cur+=1
        return array

# 方法三 推荐 双指针方法
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] & 0x01:
                i += 1
            while i < j and not nums[j] & 0x01:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums

'''
18.链表中倒数第k个结点
！！注意不是输出第k个结点的值而是第k个结点！！
'''
# 可以用快慢指针去做
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        pre, cur = head, head
        while k:
            cur = cur.next
            k-=1
        while cur:
            pre, cur = pre.next, cur.next
        return pre

'''
19.反转链表
'''
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head: return head
        pre, cur = None, head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp
        return pre
'''
20.合并两个排序的链表
'''
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val>l2.val:
                l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1 or l2

'''
21.链表中环的入口节点
'''
class Solution:
    def EntryNodeOfLoop(self, pHead):
        pre, cur = pHead, pHead
        flag = False
        while cur and pre:
            pre = pre.next
            if cur.next:
                cur = cur.next.next
                if pre == cur:
                    flag = True
                    pre = pre.next
                    n = 1
                    while pre != cur:
                        pre = pre.next
                        n += 1
                    break
            else:
                cur = cur.next
        if not flag:
            return
        pre, cur = pHead, pHead
        while n:
            cur = cur.next
            n -= 1
        while pre != cur:
            pre = pre.next
            cur = cur.next
        return pre

'''
22.树的子结构
'''
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        else:
            # 匹配到与pRoot2相同的根节点
            if pRoot1.val == pRoot2.val:
                # 每次重新遍历是不是有点耗时
                if self.equal(pRoot1, pRoot2): return True
            if self.HasSubtree(pRoot1.left, pRoot2): return True
            if self.HasSubtree(pRoot1.right, pRoot2): return True

        # 根据pRoot2要么是左树的子树，要么是右树的子树，前面代码也可以这么写：
        '''
        if not pRoot1 or not pRoot2:
            return False
        else:
            # 匹配到与pRoot2相同的根节点 
            return self.equal(pRoot1, pRoot2)|self.equal(pRoot1.left, pRoot2)|self.equal(pRoot1.right, pRoot2)
        '''
    def equal(self, node1, node2):
        if not node2:
            return True
        if not node1:
            return False
        if node1.val != node2.val:
            return False
        return self.equal(node1.left, node2.left) and self.equal(node1.right, node2.right)

'''
23.二叉树的镜像
代码为递归做法，如果要求不能用递归，可以用广度遍历去做（题解的思路
'''
# 递归实现
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root

# 循环实现
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, pRoot):
        if not pRoot:
            return True
        else:
            tmp = [pRoot]
        tmp1 = []
        while tmp:
            node = tmp.pop()
            node.left, node.right = node.right, node.left
            if node.left:
                tmp1.append(node.left)
            if node.right:
                tmp1.append(node.right)
            if not tmp:
                tmp = tmp1
                tmp1 = []
        return pRoot

'''
24.对称的二叉树
'''
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def helper(root1, root2):
            if not root1 and not root2:
                return True
            if not root1 or not root2:
                return False
            if root1.val != root2.val:
                return False
            return helper(root1.left, root2.right) and helper(root1.right, root2.left)
        return helper(root.left, root.right) if root else True

'''
25.顺时针打印矩阵
'''
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        dire = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        if not matrix: return res
        m = len(matrix)
        n = len(matrix[0])
        marked = [[0]*n for _ in range(m)]
        i, j, k = 0, 0, 0
        while 0<=i<m and 0<=j<n and not marked[i][j]:
            res.append(matrix[i][j])
            marked[i][j] = 1
            while 0<=i+dire[k][0]<m and 0<=j+dire[k][1]<n and not marked[i+dire[k][0]][j+dire[k][1]]:
                i, j = i+dire[k][0], j+dire[k][1]
                res.append(matrix[i][j])
                matrix[i][j] = 1
            k = (k+1)%4
            i, j = i+dire[k][0], j+dire[k][1]
        return res

'''
26.包含min函数的栈
'''
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.helper = []
        self.minnum = None

    def push(self, x: int) -> None:
        self.stack.append(x)
        if self.minnum == None or x<self.minnum:
            self.minnum = x
        self.helper.append(self.minnum)

    def pop(self) -> None:
        if self.stack:
            self.stack.pop()
            self.helper.pop()
        else:
            return
        if self.stack:
            self.minnum = self.helper[-1]
        else:
            self.minnum = None

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.minnum

'''
27.栈的压入、弹出序列
'''
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        res = []
        idx = 0
        for i in pushed:
            while res and res[-1] == popped[idx]:
                idx+=1
                res.pop()
            res.append(i)
        # print(res, popped)
        if res == popped[idx:][::-1]:
            return True
        else:
            return False

'''
28.从上到下打印二叉树
'''
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root: return []
        res, tmp = [], []
        stack = [root]
        while stack:
            node = stack.pop(0)
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res

'''
29.二叉搜索树的后序遍历
'''
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if not postorder:
            return True
        else:
            tmp = []
            for i in range(len(postorder)-1):
                if postorder[i] < postorder[-1]:
                    tmp.append(postorder[i])
                else:
                    break
            for j in range(len(tmp), len(postorder)-1):
                if postorder[j]<postorder[-1]:
                    return False
            return self.verifyPostorder(tmp) and self.verifyPostorder(postorder[len(tmp): len(postorder)-1])

'''
30.二叉树中和为某一值的路径
'''
class Solution:
    def pathSum(self, root: TreeNode, sumr: int) -> List[List[int]]:
        res = []
        def dfs(node, tmp):
            if sum(tmp)+node.val == sumr and not node.left and not node.right:
                res.append(tmp+[node.val])
            else:
                if node.left:
                    dfs(node.left, tmp+[node.val])
                if node.right:
                    dfs(node.right, tmp+[node.val])
        if root:
            dfs(root, [])
        return res

'''
31.复杂链表的复制
'''
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        lookup = {}
        def helper(node):
            if not node:
                return None
            if node in lookup: return lookup[node]
            copynode = Node(node.val)
            lookup[node] = copynode
            copynode.next, copynode.random = helper(node.next), helper(node.random)
            return copynode
        return helper(head)

#方法二
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None
        node = head
        while node:
            node.next, node.next.next = Node(node.val), node.next
            node = node.next.next
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

'''
32.二叉树与双向链表
'''
# 方法一 按照中序遍历保存节点然后连线
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root: return None
        array = []
        self.inorder(root, array)
        for i in range(len(array)):
            array[i].left = array[i-1]
            array[i].right = array[(i+1)%(len(array))]
        return array[0]

    def inorder(self, node, array):
        if not node:
            return
        self.inorder(node.left, array)
        array.append(node)
        self.inorder(node.right, array)

# 递归方法
class Solution:
    def __init__(self):
        self.pre = None
        self.head = None
        # self.tail = None

    def treeToDoublyList(self, pRootOfTree: 'Node') -> 'Node':
        if not pRootOfTree:
            return None
        self.inorder(pRootOfTree)
        self.head.left = self.pre
        self.pre.right = self.head
        return self.head

    def inorder(self, node):
        if not node:
            return
        self.inorder(node.left)
        if not self.pre:
            self.head = node
        else:
            self.pre.right = node
            node.left = self.pre
        self.pre = node
        # self.tail = node
        self.inorder(node.right)

'''
33.字符串的排列
'''
class Solution:
    def permutation(self, s: str) -> List[str]:
        res = []
        s = sorted(s)
        def helper(s, tmp):
            if not s:
                res.append(''.join(tmp))
            else:
                for i in range(len(s)):
                    if i>0 and s[i-1] == s[i]:
                        continue
                    helper(s[:i]+s[i+1:], tmp+[s[i]])
        helper(s,[])
        return res

# 来个内置函数的用法
import itertools
class Solution:
    def permutation(self, s: str) -> List[str]:
        res = itertools.permutations(s, len(s))
        return set([''.join(x) for x in res])

# 不用递归 交换元素
class Solution:
    def Permutation(self, s):
        # write code here
        if not s: return []
        s = list(s)
        res = [s]
        start = 0
        n = len(s)
        while start < n:
            tmp = []
            for i in res:
                for j in range(start+1, n):
                    if i[start] != i[j]:
                        tmp1 = i[:]
                        tmp1[start], tmp1[j] = tmp1[j], tmp1[start]
                        tmp.append(tmp1)
            res += tmp
            start += 1
        return sorted(list(set([''.join(x) for x in res])))

'''
34.数组中出现次数超过一半的数字
主要方法：哈希、摩尔投票、排序
'''
# 摩尔投票
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cand = None
        count = 0
        for i in nums:
            if count == 0:
                cand = i
            cand += (1 if cand == i else -1)
        return cand

# 拓展 数组中超过三分之一的元素 leecode229

'''
35.最小的k个数
'''
# 方法一：快排
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        left, right = 0, len(arr)-1
        def helper(arr, left, right):
            if left<right:
                mid = self.quicksort(arr, left, right)
                if mid>k:
                    helper(arr, left, mid-1)
                if mid<k:
                    helper(arr, mid+1, right)
        helper(arr, left, right)
        return arr[:k]

    def quicksort(self, arr, left, right):
        base = left
        i, idx = left+1, left+1
        while i<=right:
            if arr[i]<arr[base]:
                arr[idx], arr[i] = arr[i], arr[idx]
                idx+=1
            i+=1
        arr[base], arr[idx-1] = arr[idx-1], arr[base]
        return idx-1

# 堆排
class Solution:
    def getLeastNumbers(self, nums: List[int], k: int) -> List[int]:
        res = []
        n = len(nums)
        if n == 1: return nums
        for i in range(n//2-1, -1, -1):   #构建小顶堆
            self.heapify(nums, i, n-1)
        for i in range(n-1, n-k-1, -1):
            res.append(nums[0])
            nums[i], nums[0] = nums[0], nums[i] #将小顶堆堆顶放到最后
            self.heapify(nums, 0, i-1)  #调整剩下的数为小顶堆
        return res

    def heapify(self, arr, par, end):
        temp = arr[par]  # 父节点值
        son = 2*par+1 #左子结点
        while son<=end:
            if son<end and arr[son]>arr[son+1]: #选择左子结点和右子结点较小的一个
                son+=1
            if temp<=arr[son]:
                break
            arr[par]=arr[son]  #若大于父节点 上浮
            par=son
            son=2*son+1
        arr[par]=temp

'''
36.连续子数组的最大和
修改了一下可以输出区间，left,right为最大和的左右区间
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxsum = nums[0]
        cursum = nums[0]
        # left, right, i_tmp = 0, 0, 0
        for i in range(1, len(nums)):
            if cursum <= 0:
                cursum = nums[i]
                # i_tmp = i
            else:
                cursum += nums[i]
            if cursum > maxsum:
                maxsum = cursum
                # left = i_tmp
                # right = i
        # 获取区间
        # print(left, right)
        return maxsum

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划，原地修改数组
        maxnum = nums[0]
        for i in range(1,len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            maxnum = max(maxnum,nums[i])
        return maxnum

# 简洁版
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur_sum = max_sum = nums[0]
        for i in range(1, len(nums)):
            cur_sum = max(nums[i], cur_sum+nums[i])
            max_sum = max(max_sum, cur_sum)
        return max_sum

'''
37.数字1的个数
讲道理我觉得这是数学题目，我也就看懂了一种方法，
'''
# 按照编程之美上的解释写出来的
class Solution:
    def countDigitOne(self, n: int) -> int:
        if n<1:
            return 0
        res = 0
        strn = str(n)
        lenn = len(strn)
        for i in range(len(strn)):
            high = n//(10**(lenn-i))
            low = n%(10**(lenn-1-i))
            if strn[i] == '0':
                res += high*(10**(lenn-1-i))
            elif strn[i] == '1':
                res += high*(10**(lenn-1-i))
                res += (low+1)
            else:
                res += (high+1)*(10**(lenn-1-i))
        return res

# 简洁版
class Solution:
    def countDigitOne(self, n: int) -> int:
        base, res, s = 1, 0, str(n)
        for i in range(len(s)-1, -1, -1):
            n //= 10
            res += n*base
            if s[i] == '1' and i!= len(s)-1:
                res += int(s[i+1:]) + 1
            elif s[i] != '0' :
                res += base
            base *= 10
        return res

# 评论大神的写法 求任意数字钟x出现的次数  @leetao
import math
class Solution:
    def countDigitOne(self, n: int) -> int:
        def count_x_between_one_and_n(n,x):
            if n < 0 or x < 1 or x > 9:
                return 0
            high,low,current,tmp,i = 1,1,1,1,1
            high = n
            total = 0
            while high !=0:
                high = int(n/int(math.pow(10,i)))
                tmp = int(n%int(math.pow(10,i)))
                current = int(tmp/int(math.pow(10,i-1)))
                low = int(tmp%int(math.pow(10,i-1)))
                if current == x:
                    total += high*int(math.pow(10,i-1))+low+1
                elif current < x:
                    total += high*int(math.pow(10,i-1))
                else:
                    total += (high+1)*int(math.pow(10, i-1))
                i+=1
            return total
        return count_x_between_one_and_n(n,1)

'''
38.数字序列中某一位的数字
'''
# 根据剑指offer的题解写出来的
class Solution:
    def findNthDigit(self, n: int) -> int:
        if n == 0: return 0
        base = 1
        i = 1   # 当前位数
        while (n-base)>=0:
            n -= base
            base = i*9*10**(i-1)
            i += 1
        num = str(10**(i-2)+n//(i-1))
        k = n%(i-1)
        return int(num[k])

'''
39.把数组排成最小的数
'''
import functools
class Solution:
    def PrintMinNumber(self, nums):
        # write code here
        strn = [str(x) for x in nums]
        strn.sort(key = functools.cmp_to_key(self.compare))
        return ''.join(strn)

    def compare(self, a, b):
        if int(a+b)>int(b+a):
            return 1
        else:
            return -1

'''
40.把数字翻译成字符串
'''
class Solution:
    def translateNum(self, num: int) -> int:
        strn = str(num)
        n = len(strn)
        res = [0]*(n-1)+[1]
        for i in range(n-1, -1, -1):
            if i<n-1:
                res[i] = res[i+1]
                tmp = int(strn[i]+strn[i+1])
                if tmp>=10 and tmp<=25:
                    if i<n-2:
                        res[i] += res[i+2]
                    else:
                        res[i] += 1
        return res[0]

'''
41.礼物的最大价值
dfs超时，用的动态规划方法
'''
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m: n = len(grid[0])
        else: return sum(grid)
        res = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                # print(res)
                if i == 0:
                    res[i][j] = res[i][j-1]+grid[i][j]
                elif j == 0:
                    res[i][j] = res[i-1][j]+grid[i][j]
                else:
                    res[i][j] = max(res[i-1][j], res[i][j-1])+grid[i][j]
        return res[m-1][n-1]
# 简化版本
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m: n = len(grid[0])
        else: return sum(grid) # 0
        res = [0]*(n+1)
        for i in range(m):
            for j in range(n):
                res[j] = max(res[j-1], res[j])+grid[i][j]
        return res[-2]

'''
42.最长不含重复字符的子字符串
'''
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        tmp = {}
        maxs = 0
        begin = 0
        for i in range(len(s)):
            if s[i] in tmp and tmp[s[i]]>=begin:
                begin = tmp[s[i]]+1
            tmp[s[i]] = i
            maxs = max(maxs, i+1-begin)
        return maxs

# 划窗做法
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = 0
        has, window = set(), []
        for i in range(len(s)):
            if s[i] in has:
                while 1:
                    tmp = window.pop(0)
                    has.remove(tmp)
                    if tmp == s[i]:
                        break
            window.append(s[i])
            has.add(s[i])
            res = max(res, len(window))  # 这里求len可以优化，res+-1
        return res

# 动态规划

'''
43.丑数
'''
# 堆 (需要处理重复丑数）
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        import heapq
        heap = [1]
        heapq.heapify(heap)
        for _ in range(n):
            tmp = heapq.heappop(heap)
            while heap and tmp==heap[0]:
                tmp = heapq.heappop(heap)
            for i in [tmp*2, tmp*3, tmp*5]:
                heapq.heappush(heap, i)
        return tmp

# 动态规划
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1 for _ in range(n)]
        # 三指针初始化
        l2, l3, l5 = 0, 0, 0
        for i in range(1,n):
            min_val = min(dp[l2]*2,dp[l3]*3,dp[l5]*5)
            dp[i] = min_val
            # 找出哪个指针对应的数造出了现在这个最小值，将指针前移一位
            if dp[l2]*2 == min_val:
                l2 += 1
            if dp[l3]*3 == min_val:
                l3 += 1
            if dp[l5]*5 == min_val:
                l5 += 1
        return dp[-1]

'''
44.第一个只出现一次的字符
哈希
'''
class Solution:
    def firstUniqChar(self, s: str) -> str:
        res = {}
        for i in s:
            res[i] = res.get(i, 0)+1
        for i in s:
            if res[i] == 1:
                return i
        return " "

'''
45.数组中的逆序对
'''
# 归并排序做法
class Solution:
    def InversePairs(self, nums):
        self.res = 0
        self.mergesort(nums)
        return self.res % 1000000007

    def mergesort(self, nums):
        if len(nums) < 2:
            return nums
        mid = (len(nums) + 1) // 2
        sub1 = self.mergesort(nums[:mid])
        sub2 = self.mergesort(nums[mid:])
        tmp = []
        l, r = 0, 0
        while l < len(sub1) or r < len(sub2):
            if r == len(sub2) or l < len(sub1) and sub1[l] <= sub2[r]:
                tmp.append(sub1[l])
                self.res += r
                l += 1
            else:
                tmp.append(sub2[r])
                r += 1
        return tmp

# 二叉树做法  好像有点问题，有时间看一下：
class tree():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.leftsum = 0
class Solution:
    def InversePairs(self, nums):
        self.res = 0
        root = tree(nums[-1])
        for i in nums[::-1]:
            self.insert(root, i)
        return self.res % 1000000007

    def insert(self, node, val):
        while node.val != val:
            if node.val > val:
                if not node.left:
                    node.left = tree(val)
                node.leftsum += 1
                node = node.left
            else:
                self.res += (node.leftsum + 1)
                if not node.right:
                    node.right = tree(val)
                node = node.right

'''
46.两个链表的第一个公共节点
用了剑指offer上最后个方案
'''
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lena, lenb = 0, 0
        tmpa, tmpb = headA, headB
        while tmpa:
            lena+=1
            tmpa=tmpa.next
        while tmpb:
            lenb+=1
            tmpb=tmpb.next
        if lena>lenb:
            longhead = headA
            shorthead = headB
            tmp = lena-lenb
        else:
            longhead = headB
            shorthead = headA
            tmp = lenb-lena
        while tmp:
            longhead = longhead.next
            tmp-=1
        while longhead and shorthead:
            if longhead == shorthead:
                return longhead
            else:
                longhead = longhead.next
                shorthead = shorthead.next
        return None

'''
47.在排序数组中查找数字I
'''
# 二分法
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        return len(nums[self.getleft(nums, target):self.getright(nums, target)])
        # print(self.getright(nums, target))

    def getleft(self, nums, target):
        l, r = 0, len(nums)-1
        while l<=r:
            mid = l+(r-l)//2
            if nums[mid]<target:
                l = mid+1
            else:
                r = mid-1
        return l

    def getright(self, nums, target):
        l, r = 0, len(nums)-1
        while l<=r:
            mid = l+(r-l)//2
            if nums[mid]<=target:
                l = mid+1
            else:
                r = mid-1
        return l

'''
48.0~n-1中缺失的数字
'''
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        l, r = 0, len(nums)-1
        while l<=r:
            mid = l+(r-l)//2
            if nums[mid] == mid:
                l = mid+1
            else:
                r = mid-1
        return l

'''
49.二叉搜索树的第K大节点
'''
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root: return
        stack = []
        count = 0
        while root or stack:
            while root:
                stack.append(root)
                root = root.right
            root = stack.pop()
            count += 1
            if count == k: return root.val
            root = root.left

'''
50.二叉树的深度
'''
# 方法一 递归
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        return 1+max(self.maxDepth(root.left), self.maxDepth(root.right))
# 方法二 stack
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        stack, tmp, depth = [root], [], 0
        while stack:
            node = stack.pop()
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
            if not stack:
                stack = tmp
                tmp = []
                depth += 1
        return depth

'''
51.平衡二叉树
'''
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        self.res = True
        def helper(root):
            if not root:
                return 0
            left = helper(root.left) + 1
            right = helper(root.right) + 1
            if abs(right - left) > 1:
                self.res = False
            return max(left, right)
        helper(root)
        return self.res

# 比上一个方法又好一点
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        return self.depth(root) != -1

    def depth(self, root):
        if not root: return 0
        left = self.depth(root.left)
        if left == -1: return -1
        right = self.depth(root.right)
        if right == -1: return -1
        return max(left, right) + 1 if abs(left - right) < 2 else -1

'''
52.数组中数字出现的次数
'''
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        res = 0  # 所有数字异或的结果
        a, b = 0, 0
        for i in nums:
            res ^= i
        h = 1
        while(res & h == 0):
            h <<= 1
        for i in nums:
            if (h & i == 0):
                a ^= i
            else:
                b ^= i
        return [a, b]

'''
53.和为s的两个数字
'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i, j = 0, len(nums)-1
        while i<j:
            if (nums[i]+nums[j]) == target:
                return [nums[i], nums[j]]
            if (nums[i]+nums[j])>target:
                j-=1
            else:
                i+=1

'''
54.和为s的整数序列
'''
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        if target < 3: return []
        n = (target+1)//2
        i, j, tmp = 1, 2, 3
        res = []
        while i < n:
            if tmp>target:
                tmp-=i
                i+=1
            else:
                if tmp == target:
                    res.append([k for k in range(i, j+1)])
                j+=1
                tmp+=j
        return res

'''
55.翻转单词顺序
'''
class Solution:
    def reverseWords(self, s: str) -> str:
        # '  abc def   ghi  '->'ghi def abc'
        res = []
        n = len(s)
        wordl, wordr = n-1, n-1
        while wordl>=0:
            while wordl>=0 and s[wordl] == ' ':
                wordl -= 1
            wordr = wordl
            while wordl>=0 and s[wordl]!=' ':
                wordl-=1
            if wordr>=0:
                res.append(s[wordl+1: wordr+1])
        return ' '.join(res)

'''
另一种翻转形式
'  abc def   ghi  '->'cba fed ihg'
'''
class Solution:
    def reverseWords(self, s: str) -> str:
        res = []
        n = len(s)
        wordl, wordr = n-1, n-1
        while wordl>=0:
            while wordl>=0 and s[wordl] == ' ':
                wordl-=1
            wordr = wordl
            tmp = ''
            while wordl>=0 and s[wordl]!=' ':
                tmp+=s[wordl]
                wordl -= 1
            if wordr>=0:
                res = [tmp]+res
        return ' '.join(res)

'''
56.反转字符串
'''

'''
57.滑动窗口最大值
'''
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if len(nums)<k:
            return []
        res, tmp = [], []
        for i, v in enumerate(nums):
            if i >= k and tmp[0] <= i - k: tmp.pop(0)
            while tmp and nums[tmp[-1]] <= v: tmp.pop()
            tmp.append(i)
            if i >= k - 1: res.append(nums[tmp[0]])
        return res

'''
58.队列中的最大值
'''
class MaxQueue:
    def __init__(self):
        self.queue = []
        self.maxque = []

    def max_value(self) -> int:
        # print(self.queue, self.maxque)
        if self.maxque:
            return self.maxque[0]
        else:
            return -1

    def push_back(self, value: int) -> None:
        self.queue.append(value)
        while self.maxque and self.maxque[-1]<value:
            self.maxque.pop()
        self.maxque.append(value)

    def pop_front(self) -> int:
        if not self.queue:
            return -1
        if self.maxque[0] == self.queue[0]:
            self.maxque.pop(0)
        return self.queue.pop(0)

'''
59.n个骰子的点数
'''
#  又是一段超时代码
class Solution:
    def twoSum(self, n: int) -> List[float]:
        self.res = [0]*(6*n-n+1)
        self.k = n
        self.helper(n, 0)
        return [x/(6**n) for x in self.res]

    def helper(self, n, sumn):
        if n==0:
            self.res[sumn-self.k] += 1
        else:
            for i in range(1, 7):
                self.helper(n-1, sumn+i)
# 动态规划法
class Solution:
    def twoSum(self, n: int) -> List[float]:
        dp = [0] * (n*6+1)
        for i in range(1, n*6+1):
            if i <= 6:
                dp[i] = 1
        for i in range(1, n):
            for j in range(n*6, 0, -1):
                start = j-6 if j-6 >= 0 else 0
                dp[j] = sum(dp[start:j])
        m=sum(dp)
        for i in range(len(dp)):
            dp[i]/=m
        return dp[-5*n-1:]

'''
60.扑克牌中的顺子
'''
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        nums.sort()
        tmp, res = 0, 0
        for i in range(len(nums)):
            if nums[i] == 0:
                tmp+=1
            else:
                if i>0 and nums[i-1]!=0:
                    tmp1 = nums[i]-nums[i-1]-1
                    res += tmp1
                    if tmp1<0:
                        return False
        if res>tmp:
            return False
        else:
            return True

'''
61.圆圈中剩下的数字
'''
# 超时代码O(mn)
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        lst = list(range(n))
        last = 0
        for i in range(n-1):
            last = (last+m-1) % len(lst)
            lst = lst[:last]+lst[last+1:]
        return lst[0]

# 数学规律
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        last = 0
        for i in range(2, n+1):
            last = (last+m)%i
        return last

'''
62.股票的最大利润
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        res = 0
        minp = 0
        for i in range(1, len(prices)):
            tmp = prices[i]-prices[minp]
            if tmp<=0:
                minp = i
            else:
                res = max(res, tmp)
        return res
# 记录买卖时刻
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        res = 0
        minp, maxp, p = 0, 0, 0
        for i in range(1, len(prices)):
            tmp = prices[i]-prices[p]
            if tmp<=0:
                p = i
            else:
                if tmp>res:
                    minp = p
                    maxp = i
                    res = max(res, tmp)
        return res

'''
63.不用加减乘除做加法
'''
# 详细解释：https://zhuanlan.zhihu.com/p/64642722
class Solution:
    def add(self, a: int, b: int) -> int:
        a = a&0xffffffff
        b = b&0xffffffff
        while b:
            tmp1 = a^b
            tmp2 = (a&b)<<1
            a = tmp1&0xffffffff
            b = tmp2&0xffffffff
        return a if a < 0x80000000 else ~(a^0xFFFFFFFF)

'''
64.求1+2+…+n
'''
class Solution:
    def sumNums(self, n: int) -> int:
        return n and n+self.sumNums(n-1)

'''
40.构建乘积数组
'''
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        tmp1 = [1]
        tmp2 = [1]
        res = []
        for i in range(1, len(a)):
            tmp1.append(tmp1[-1]*a[i-1])
            tmp2.append(tmp2[-1]*a[-i])
        for j in range(len(a)):
            res.append(tmp1[j]*tmp2[len(a)-1-j])
        return res
# 题解
# 这种对称扫描方法比较实用
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        res = [1 for i in range(len(a))]
        left = 1
        for i in range(len(a)):
            res[i] = left
            left *= a[i]
        right = 1
        for i in range(len(a)-1,-1,-1):
            res[i] *= right
            right *= a[i]
        return res





# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       剑指offer
   Description :
   Author :         amilyxy
   date：           2020/1/23
-------------------------------------------------
"""
'''
1.二维数组中的查找
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

# 优化方案一 ： 左上寻找 逐步减少查询范围   时间复杂度最多能O(n^2),好像并没有很优化多少
class Solution:
    def Find(self, target, array):
        m = len(array)
        n = len(array[0])
        temp = 0
        while temp<m and temp<n:
            for i in range(temp, m):
                if array[i][temp] == target:
                    return True
                if array[i][temp] > target:
                    m = i
                    break
            for j in range(temp, n):
                if array[temp][j] == target:
                    return True
                if array[temp][j] > target:
                    n = j
                    break
            temp+=1
        return False

# 优化方案二 右上查询 时间复杂度O(m+n)
class Solution:
    def Find(self, target, array):
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
2. 替换空格
'''
class Solution:
    def replaceSpace(self, s):
        s = list(s)
        for i in range(len(s)-1, -1, -1):
            if s[i] == ' ':
                s[i] = '%20'
        return ''.join(s)

'''
3. 从尾到头打印链表
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
4. 重建二叉树
'''
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
4.二叉树的下一个节点
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
5.用两个栈实现队列
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
        return self.stack2.pop()

'''
6.旋转排序数组中的最小值
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

'''
7.菲波那切数列
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
8.矩阵中的路径
力扣中相似题目有8、90、77、39
'''
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if len(path) == 0: return True
        dire = [[-1, 0], [0, -1], [1, 0], [0, 1]]

        def helper(row, col, path, marked):
            marked[row][col] = 1
            if len(path) == 0: return True
            for i in dire:
                newi, newj = row + i[0], col + i[1]
                if 0 <= newi < rows and 0 <= newj < cols and matrix[newi * cols + newj] == path[0] and marked[newi][newj] == 0:
                    if helper(newi, newj, path[1:], marked): return True

        for i in range(rows):
            for j in range(cols):
                if matrix[i * cols + j] == path[0]:
                    marked = [[0 for _ in range(cols)] for _ in range(rows)]
                    if helper(i, j, path[1:], marked): return True
        return False

'''
9. 机器人的运动范围
'''
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        dire = [[-1, 0], [0, 1], [1, 0], [0, 1]]
        marked = [[0 for _ in range(cols)] for _ in range(rows)]

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
                if 0 <= newi < rows and 0 <= newj < cols and marked[newi][newj] == 0:
                    if bitsum(newi) + bitsum(newj) <= threshold:
                        helper(newi, newj)

        if threshold > 0: helper(0, 0)
        return sum([sum(i) for i in marked])

'''
10.剪绳子
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
11.数值的整数次方
'''
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0 and exponent<0:
            return "the input is invalid"
        else:
            return pow(base, exponent)  # base**exponent

'''
12.调整数组顺序使得奇数位于偶数前面
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

'''
13.链表中倒数第k个结点
！！注意不是输出第k个结点的值而是第k个结点！！
'''
# 可以用快慢指针去做
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        slow, fast = head, head
        while k:
            if fast:
                fast = fast.next
            else:
                return None
            k = k-1
        while fast:
            fast = fast.next
            slow = slow.next
        return slow

'''
14.树的子结构
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
15.数组中的逆序对
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

# 二叉树做法：
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
16.二叉树的镜像  
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
17.栈的压入、弹出序列
'''
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        stack = [pushV[0]]
        pushV = pushV[1:]
        while stack:
            if stack[-1] != popV[0]:
                if pushV:
                    stack.append(pushV.pop(0))
                else:
                    return False
            else:
                stack.pop()
                popV.pop(0)
        return True

'''
18.从上到下打印二叉树
'''
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        res, tmp = [], []
        if root:
            stack = [root]
        else:
            return res
        while stack:
            node = stack.pop(0)
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res

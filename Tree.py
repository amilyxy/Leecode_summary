# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Tre
   Description :
   Author :         amilyxy
   date：           2019/9/4
-------------------------------------------------
"""
'''
144. Binary Tree Preorder Traversal: 二叉树的前序遍历
describe: 给定一个二叉树，返回它的前序遍历。
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 方法一: 题解方法
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        list = []
        #递归思路操作 先根节点 ->左节点->右节点
        if root is None:   # 基准条件
            return []
        # 调换下面这三个的顺序就可以实现前中后序遍历
        list.append(root.val)                           #这里不能用extend
        list.extend(self.preorderTraversal(root.left))  #这里不能用append
        list.extend(self.preorderTraversal(root.right))
        return list

    # 方法二：基于栈的实现：先进后出
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        stack, result = [], []
        if root:
            stack.append(root)
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result

    # 方法三 莫里斯遍历
    class Solution(object):
        def preorderTraversal(self, root):
            """
            :type root: TreeNode
            :rtype: List[int]
            """
            node, output = root, []
            while node:
                if not node.left:
                    output.append(node.val)
                    node = node.right
                else:
                    predecessor = node.left

                    while predecessor.right and predecessor.right is not node:
                        predecessor = predecessor.right

                    if not predecessor.right:
                        output.append(node.val)
                        predecessor.right = node
                        node = node.left
                    else:
                        predecessor.right = None
                        node = node.right

            return output
'''
94. Binary Tree Inorder Traversal: 二叉树的中序遍历
describe: 给定一个二叉树，返回它的中序 遍历。
'''
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 方法一 递归
        # out = []
        # if root == None:
        #     return []
        # out.extend(self.inorderTraversal(root.left))
        # out.append(root.val)
        # out.extend(self.inorderTraversal(root.right))
        # return out

        # 方法二 堆栈的方法 空间复杂度相当的高
        stack, out = [], []
        if root:
            stack = [root]
        while stack:
            node = stack.pop()
            if type(node) == int:
                out.append(node)
                continue
            if node.right:
                stack.append(node.right)
            stack.append(node.val)
            if node.left:
                stack.append(node.left)
        return out

        # 评论方法 这个方法很不错！！
        ret, st, n = [], [], root
        while n or st:
            while n:
                st.append(n)
                n = n.left
            n = st.pop()
            ret.append(n.val)
            n = n.right
        return ret

'''
145. Binary Tree Postorder Traversal: 二叉树的后序遍历
describe: 给定一个二叉树，返回它的后序遍历。
'''
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        # 方法一 递归
        # out = []
        # if root == None:
        #     return []
        # out.extend(self.postorderTraversal(root.left))
        # out.extend(self.postorderTraversal(root.right))
        # out.append(root.val)
        # return out

        # 方法二 迭代 空间复杂度相当的高
        stack, out = [], []
        if root:
            stack = [root]
        while stack:
            node = stack.pop()
            if type(node) == int:
                out.append(node)
                continue
            stack.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return out

'''
102. Binary Tree Level Order Traversal 二叉树的层次遍历
'''
# BFS方法
class Solution:
    def levelOrder(self, root: TreeNode) -> list[list[int]]:
        tree, temp, res, treedown = [], [], [], []
        if root:
            tree.append(root)
        while tree:
            node = tree.pop(0)
            temp.append(node.val)
            if node.left:
                treedown.append(node.left)
            if node.right:
                treedown.append(node.right)
            if not tree:
                res.append(temp)
                tree = treedown
                treedown, temp = [], []
        return res

# DFS方法
class Solution:
    def levelOrder(self, root: TreeNode) -> list[list[int]]:
        res = []
        def _DFS(node, depth):
            if not node: return
            if len(res) == depth:
                res.append([])
            res[depth].append(node.val)
            _DFS(node.left, depth+1)
            _DFS(node.right, depth+1)
        _DFS(root, 0)
        return res

'''
100. same tree 相同的树
'''
# 方法一： 如果两个树相等 他们遍历结果必定也是相同的
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # 采用先序遍历
        resp = self.traversal(p)
        resq = self.traversal(q)
        print(resp)
        print(resq)
        if resp == resq:
            return True
        else:
            return False

    def traversal(self, node):
        res = []
        if node:
            res.append(node.val)
        else:
            res.append('#')
            return res
        res.extend(self.traversal(node.left))
        res.extend(self.traversal(node.right))
        return res

# 层次遍历方法也可以做呐
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        # 采用层次遍历
        a, b = self.helper(p), self.helper(q)
        if a == b:
            return True
        else:
            return False

    def helper(self, node):
        temp, res = [], []
        if node:
            listnode = [node]
            while listnode:
                n = listnode.pop()
                if n == '#':
                    res.append('#')
                else:
                    res.append(n.val)
                    temp.append(n.left) if n.left else temp.append('#')
                    temp.append(n.right) if n.right else temp.append('#')
                if not listnode:
                    listnode = temp
                    temp = []
        return res

# 题解方法 递归
'''
做剑指offer的时候发现这种方法比前面两个好太多了，还用’#‘代替空节点，万一树本身就有’#‘呢
'''
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        # one of p and q is None
        if not q or not p:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)

'''
101 Symmetric Tree 对称二叉树
'''
# 突然觉得 用我上面的自己层次遍历 最后判断每一层res是否对称就好了
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root:
            listnode = [root]
            temp, res = [], []
            while listnode:
                n = listnode.pop()
                if n == '#':
                    res.append('#')
                else:
                    res.append(n.val)
                    temp.append(n.left) if n.left else temp.append('#')
                    temp.append(n.right) if n.right else temp.append('#')
                if not listnode:
                    for i in range(len(res)//2):
                        if res[i] != res[len(res)-i-1]:
                            return False
                    listnode = temp
                    temp, res = [], []
        return True

# 递归方法思考
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        return self.helper(root, root)

    def helper(self, node1, node2):
        if node1 and node2:
            if node1.val != node2.val:
                return False
            else:
                return self.helper(node1.left, node2.right) and self.helper(node1.right, node2.left)
        else:
            if node1 or node2:
                return False
            else:
                return True
'''
572.另一个树的子树
'''
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s or not t:
            return False
        if s.val == t.val:
            if self.issame(s, t): return True
        if self.isSubtree(s.left, t): return True
        if self.isSubtree(s.right, t): return True

    def issame(self, s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        if s.val != t.val:
            return False
        return self.issame(s.left, t.left) and self.issame(s.right, t.right)

'''
113.路径总和
'''
# 刚开始自己写的dfs方法 但我觉得我写的好像有点复杂，应该题解有更简洁的写法
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
199.二叉树的右视图
'''
# 方法一 层次遍历
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        res = []
        stack = [root]
        if not root:
            return res
        next_s = []
        res.append(root.val)
        while stack:
            node = stack.pop(0)
            if node.left:
                next_s.append(node.left)
            if node.right:
                next_s.append(node.right)
            if not stack:
                stack = next_s
                if next_s:
                    res.append(next_s[-1].val)
                next_s = []
        return res






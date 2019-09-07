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










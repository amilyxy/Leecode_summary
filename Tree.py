e# -*- coding: utf-8 -*-
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












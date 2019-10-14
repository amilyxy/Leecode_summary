# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       graph
   Description :
   Author :         amilyxy
   date：           2019/10/12
-------------------------------------------------
"""
# 我的思路：  DFS + dict保留已经遍历过的节点
class Node:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        marked = {}
        newnode = Node(node.val, [])
        self.dfs(node, marked, newnode)
        return newnode

    def dfs(self, node, marked, newnode):
        # newnode.val = node.val
        marked[node] = newnode
        for nei in node.neighbors:
            # print(nei.val)
            if nei not in marked:
                temp = Node(nei.val, [])
                newnode.neighbors.append(temp)
                self.dfs(nei, marked, newnode.neighbors[-1])
            else:
                newnode.neighbors.append(marked[nei])

# 题解DFS方案： 感想就是我的好像写的乱七八糟, 题解还是牛皮的
# 从我的前一个改进答案 还是比较容易过渡到题解答案的
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        lookup = {}
        def dfs(node):
            if not node: return
            if node in lookup:
                return lookup[node]
            clone = Node(node.val, [])
            lookup[node] = clone
            for n in node.neighbors:
                clone.neighbors.append(dfs(n))
            return clone
        return dfs(node)

# 题解BFS方案：
from collections import deque
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        queue = deque()
        marked = {}
        newnode = Node(node.val, [])
        marked[node] = newnode
        queue.append(node)
        while queue:
            curnode = queue.popleft()
            for nei in curnode.neighbors:
                if nei not in marked:
                    marked[nei] = Node(nei.val, [])
                    queue.append(nei)
                marked[curnode].neighbors.append(marked[nei])
        return newnode













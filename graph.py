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

'''
399. Evaluate Division 除法求值
'''
# 先构建数据图 然后DFS遍历
# 题外话： 感觉我写的好复杂？？？
class Solution:
    def calcEquation(self, equations: list[List[str]], values: list[float], queries: list[list[str]]) -> list[
        float]:
        datagraph = {}
        n = len(values)
        # 构建datagraph
        for i in range(n):
            if equations[i][0] not in datagraph:
                datagraph[equations[i][0]] = [[equations[i][1], values[i]]]
            else:
                for j in datagraph[equations[i][0]]:
                    if equations[i][1] not in j:
                        datagraph[equations[i][0]].append([equations[i][1], values[i]])

            if equations[i][1] not in datagraph:
                datagraph[equations[i][1]] = [[equations[i][0], 1. / values[i]]]
            else:
                for j in datagraph[equations[i][1]]:
                    if equations[i][0] not in j:
                        datagraph[equations[i][1]].append([equations[i][0], 1. / values[i]])
        print(datagraph)
        res = []
        for query in queries:
            marked = []
            # if query[0] not in datagraph:
            #     res.append(-1)
            # else:
            res.append(self.dfs(query[0], query[1], datagraph, marked))
        return res

    def dfs(self, a, b, datagraph, marked):
        if a not in datagraph:
            return -1
        for j in datagraph[a]:
            if j[0] == b:
                return j[1]
            elif [a, j[0]] not in marked:
                marked.append([a, j[0]])
                temp = self.dfs(j[0], b, datagraph, marked)
                if temp != -1:
                    return j[1] * temp
        return -1













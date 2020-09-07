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
    def calcEquation(self, equations: list[List[str]], values: list[float], queries: list[list[str]]) -> list[float]:
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

'''
310 Minimum Height Trees 最小高度的树
'''
# 给我超时的答案一波画面
from collections import defaultdict
from collections import deque
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        res = {}
        dictdata = defaultdict(set)
        for edge in edges:
            dictdata[edge[0]].add(edge[1])
            dictdata[edge[1]].add(edge[0])
        # print(dictdata)
        for i in range(n):
            marked = [0 for _ in range(n)]
            depth = 0
            queue = deque()
            queue.append(i)
            while queue:
                dep = len(queue)
                for _ in range(dep):
                    a = queue.popleft()
                    marked[a] = 1
                    for k in dictdata[a]:
                        if not marked[k]:
                            queue.append(k)
                depth += 1
            res[i] = depth-1
        out = set()
        a = min([res[i] for i in res])
        for key in res:
            if res[key] == a:
                out.add(key)
        return out

# 题解方法一： @typingMonkey
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        res = {}
        dictdata = defaultdict(set)
        for edge in edges:
            dictdata[edge[0]].add(edge[1])
            dictdata[edge[1]].add(edge[0])
        # print(dictdata)
        indegree = [i for i in dictdata if len(dictdata[i])==1 ]
        if n == 1:
            return [0]
        while n > 2:
            t = set()
            for i in indegree:
                a = dictdata[i].pop()
                dictdata[a].remove(i)
                if len(dictdata[a]) == 1:
                    t.add(a)
                n -= 1
            indegree = t
        return indegree


'''
149. 直线上最多的点数
'''
# @powcai
from collections import Counter, defaultdict
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        points_dict = Counter(tuple(point) for point in points)
        no_repeat_point = list(points_dict.keys())
        n = len(no_repeat_point)
        if n == 1: return points_dict[no_repeat_point[0]]
        res = 0

        # 最大公约数的模板函数
        def gcd(a,b):
            if b!=0:
                return gcd(b,a%b)
            else:
                return a

        for i in range(1, n):
            x1, y1 = no_repeat_point[i][0], no_repeat_point[i][1]
            slope = defaultdict(int)
            for j in range(i):
                dy = y1-no_repeat_point[j][1]
                dx = x1-no_repeat_point[j][0]
                # 求斜率k
                '''
                if dx == 0:
                    tmp = '#'
                else:
                    tmp = dy/dx
                    # tmp = dy*1000/dx*1000 为了避免float的精度问题
                slope[tmp] += points_dict[no_repeat_point[j]]
                '''
                # 求最大公约数
                g = gcd(dy, dx)
                if g!=0:
                    dy //=g
                    dx //=g
                slope["{}/{}".format(dy, dx)] += points_dict[no_repeat_point[j]]

            res = max(res, max(slope.values()) + points_dict[no_repeat_point[i]])
        return res










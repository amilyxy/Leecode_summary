# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Topological_Sort
   Description :
   Author :         amilyxy
   date：           2019/10/9
-------------------------------------------------
"""
'''
207. Course Schedule 课程表
'''
# DFS 方法：@krahets jyd
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        def dfs(i, adjacency, flags):
            if flags[i] == -1: return True
            if flags[i] == 1: return False
            flags[i] = 1
            for j in adjacency[i]:
                if not dfs(j, adjacency, flags): return False
            flags[i] = -1
            return True

        adjacency = [[] for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]
        for cur, pre in prerequisites:
            adjacency[pre].append(cur)
        for i in range(numCourses):
            if not dfs(i, adjacency, flags): return False
        return True

# 拓扑排序方法 @liweiwei1419
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        classli = [[] for _ in range(numCourses)]
        classin = [0 for _ in range(numCourses)]
        for pos, pre in prerequisites:
            classli[pre].append(pos)
            classin[pos] += 1
        queue = []
        for i in range(numCourses):
            if classin[i] == 0:
                queue.append(i)

        count = 0
        while queue:
            i = queue.pop()
            count += 1
            for j in classli[i]:
                classin[j] -= 1
                if classin[j] == 0:
                    queue.append(j)
        return count == numCourses

'''
210. Course ScheduleII： 课程表II
'''
# 拓扑排序方法
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        classli = [[] for _ in range(numCourses)]
        classin = [0 for _ in range(numCourses)]
        for pos, pre in prerequisites:
            classli[pre].append(pos)
            classin[pos] += 1
        queue = []
        for i in range(numCourses):
            if classin[i] == 0:
                queue.append(i)

        res= []
        while queue:
            i = queue.pop()
            res.append(i)
            for j in classli[i]:
                classin[j] -= 1
                if classin[j] == 0:
                    queue.append(j)
        if len(res) == numCourses:
            return []
        return res
 # DFS 方法
class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        def dfs(i, classli, flags):
            if flags[i] == -1: return True
            if flags[i] == 1: return False
            flags[i] = 1
            for j in classli[i]:
                if not dfs(j, classli, flags): return False
            res.append(i)
            flags[i] = -1
            return True

        res = []
        classli = [[] for _ in range(numCourses)]
        classin = [0 for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]
        begin = []
        if len(prerequisites) == 0:
            return [i for i in range(numCourses)]
        for cur, pre in prerequisites:
            classli[pre].append(cur)
            classin[cur] += 1
        for i in range(numCourses):
            if classin[i] == 0:
                begin.append(i)
        for i in begin:
            if not dfs(i, classli, flags): return []
        if len(res) == numCourses:
            return res[::-1]
        else:
            return []








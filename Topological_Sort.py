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


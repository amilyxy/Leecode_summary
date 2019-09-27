# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       DFS&BFS
   Description :
   Author :         amilyxy
   date：           2019/9/26
-------------------------------------------------
"""
'''
200. Number of Islands: 岛屿数量
describe: 给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。
          一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。
          你可以假设网格的四个边均被水包围。
feeling: 这谁顶得住啊！！我看了半个小时，想了半个小时，念了半个小时！(紫薇语气）还是不会做！
'''
# 题解答案之DFS
class Solution:
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def numIslands(self, grid: list[list[str]]) -> int:
        m = len(grid)  # 行
        if m == 0:
            return 0
        n = len(grid[0])  # 列
        marked = [[0 for _ in range(n)] for _ in range(m)]
        islands = 0
        for i in range(m):
            for j in range(n):
                if not marked[i][j] and grid[i][j] == '1':
                    islands += 1
                    self.__dfs(grid, i, j, m, n, marked)
        return islands

    def __dfs(self, grid, i, j, m, n, marked):
        marked[i][j] = 1
        for direction in self.directions:
            new_i = i + direction[0]
            new_j = j + direction[1]
            if 0 <= new_i < m and 0 <= new_j < n and not marked[new_i][new_j] and grid[new_i][new_j] == '1':
                self.__dfs(grid, new_i, new_j, m, n, marked)

# 题解答案之BFS
from collections import deque
class Solution:
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def numIslands(self, grid: list[list[str]]) -> int:
        m = len(grid)    # 行
        if m == 0:
            return 0
        n = len(grid[0]) # 列
        marked = [[0 for _ in range(n)] for _ in range(m)]
        islands = 0
        for i in range(m):
            for j in range(n):
                if marked[i][j] == 0 and grid[i][j] == '1':
                    islands += 1
                    queue = deque()
                    queue.append((i, j))
                    marked[i][j] = 1
                    while queue:
                        cur_x, cur_y = queue.popleft()
                        for direction in self.directions:
                            new_i = cur_x + direction[0]
                            new_j = cur_y + direction[1]
                            if 0<=new_i<m and 0<=new_j<n and not marked[new_i][new_j] and grid[new_i][new_j] == '1':
                                queue.append((new_i, new_j))
                                marked[new_i][new_j] = 1
        return islands
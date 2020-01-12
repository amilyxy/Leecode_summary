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

'''
130. Surrounded Regions: 被围绕的区域
describe: 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
          找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
feeling: 啊啊啊终于做出来了
'''
from collections import deque
# BFS方法
class Solution:
    def solve(self, board: list[list[str]]) -> None:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        m = len(board)
        if m == 0:
            return 0
        n = len(board[0])
        marked = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O' and not marked[i][j]:
                    queue = deque()
                    change = deque()
                    queue.append((i, j))
                    change.append((i, j))
                    marked[i][j] == 1
                    flag = 0
                    while queue:
                        x, y = queue.popleft()
                        # 这里pop()也可以呐
                        if x == 0 or y == 0 or x == (m - 1) or y == (n - 1):
                            flag = 1
                        for dire in directions:
                            new_x = x + dire[0]
                            new_y = y + dire[1]
                            if (0 <= new_x < m) and (0 <= new_y < n) and not marked[new_x][new_y] and board[new_x][
                                new_y] == 'O':
                                queue.append((new_x, new_y))
                                change.append((new_x, new_y))
                                marked[new_x][new_y] = 1
                    if flag == 0:
                        self.changex2o(change, board)

    def changex2o(self, change, obj):
        while change:
            x, y = change.pop()
            obj[x][y] = 'X'

# DFS方法 这个真的想了好久我佛了 还是写的有点不熟练 多多练习！！！
class Solution:
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    flag = 0
    change = deque()
    def solve(self, board: list[list[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        if m == 0:
            return 0
        n = len(board[0])
        marked = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O' and not marked[i][j]:
                    self.flag = 0
                    self.dfs(board, marked, i, j, m, n)
                    if self.flag == 0:
                        self.changex2o(self.change, board)
                    self.change.clear()

    def dfs(self, board, marked, i, j, m, n):
        if i == 0 or j == 0 or i == (m - 1) or j == (n - 1):
            self.flag = 1
        marked[i][j] = 1
        self.change.append((i, j))
        for dire in self.directions:
            new_i = i + dire[0]
            new_j = j + dire[1]
            if (0 <= new_i < m) and (0 <= new_j < n) and not marked[new_i][new_j] and board[new_i][new_j] == 'O':
                self.dfs(board, marked, new_i, new_j, m, n)

    def changex2o(self, change, obj):
        while change:
            x, y = change.pop()
            obj[x][y] = 'X'

# 题解方法：


'''
127. Word Ladder: 单词接龙
describe: 给定两个单词（beginWord 和 endWord）和一个字典
          找到从 beginWord 到 endWord 的最短转换序列的长度。
feeling: 超时使人绝望
'''
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        # # 超时超时烦死了
        queue = set()
        queue.add(beginWord)
        res = 0
        n = len(beginWord)
        # 首先我真的是百思不得其解 为什么要对set(wordList) 删掉这行就超时
        # 应该是跟set和list操作的时间复杂度有关 具体简书里对应题目有写
        wordList = set(wordList)
        while queue:
            queue_next = set()
            # 获取一个字符不同的单词
            for cur in queue:
                for i in range(n):
                    for a in range(97, 123):
                        st = cur[:i] + chr(a) + cur[i+1:]
                        # 时间复杂度为O(1) 而list为O(n)
                        if st in wordList:
                            if st == endWord:
                                return res + 2
                            queue_next.add(st)
                            wordList.remove(st)
            res += 1
            queue = queue_next
        return 0

    # 题解方法-双向bfs @powcai
    # 该方法每次从中间结果较少的一端bfs，剪枝掉了许多不必要的搜索过程
    # 通过对比官方题解还是觉得这个比较简洁
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        if endWord not in wordList:
            return 0
        wordict = set(wordList)
        s1 = {beginWord}
        s2 = {endWord}
        n = len(beginWord)
        step = 0
        wordict.remove(endWord)
        while s1 and s2:
            step += 1
            if len(s1)>len(s2):
                s1, s2 = s2, s1
            s = set()
            for word in s1:
                nextword = [word[:i] + chr(a) + word[i+1:] for a in range(97, 123) for i in range(n)]
                for w in nextword:
                    if w in s2:
                        return step+1
                    if w not in wordict:
                        continue
                    wordict.remove(w)
                    s.add(w)
            # print(s)
            s1 = s
        return 0

'''
51. N-Queens N皇后
'''
# 回溯法 简洁代码 @liweiwei1419
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # col, mdiagonal, sdiagonal记录攻击位置
        res = []

        def dfs(row, col, mdiagonal, sdiagonal, cur_res):
            if row == n:
                res.append(["." * cur + "Q" + "." * (n - cur - 1) for cur in cur_res])
                return

            for i in range(n):
                if (i not in col) and (i + row not in mdiagonal) and (i - row not in sdiagonal):
                    dfs(row + 1, col | {i}, mdiagonal | {i + row}, sdiagonal | {i - row}, cur_res + [i])

        dfs(0, set(), set(), set(), [])
        return res

'''
52. N皇后II  
'''
# 其实就是51题啦
class Solution:
    def totalNQueens(self, n: int) -> int:
        global resnum
        resnum = 0
        def dfs(col, row, mdiagonal, sdiagonal):
            if col == n:
                global resnum
                resnum += 1
                return
            for i in range(n):
                if i not in row and col+i not in mdiagonal and col-i not in sdiagonal:
                    dfs(col+1, row|{i}, mdiagonal|{col+i}, sdiagonal|{col-i})

        dfs(0, set(), set(), set())
        return resnum












# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Trie
   Description :
   Author :         amilyxy
   date：           2019/10/15
-------------------------------------------------
"""
'''
211. Add and Search Word - Data structure design 添加与搜索单词 - 数据结构设计
'''
from collections import defaultdict
class WordDictionary:
    def __init__(self):
        self.data = defaultdict(set)

    def addWord(self, word: str) -> None:
        self.data[len(word)].add(word)

    def search(self, word: str) -> bool:
        has = []
        for i, j in enumerate(word):
            if j != '.':
                has.append(i)

        for w in self.data[len(word)]:
            if len(has) == 0:
                return True
            if len(w) == len(word):
                res = 0
                for k in has:
                    if word[k] == w[k]:
                        res += 1
                    else:
                        break
                if res == len(has):
                    return True
        return False

'''
208. Implement Trie (Prefix Tree)：实现 Trie (前缀树)
'''
from collections import defaultdict
class Trie:
    def __init__(self):
        self.data = defaultdict(dict)

    def insert(self, word: str) -> None:
        data = self.data
        # if self.search(word):
        #     return False
        for w in word:
            if w not in data:
                data[w] = {}
            data = data[w]
        data['end'] = {1}
        # return True

    def search(self, word: str) -> bool:
        data = self.data
        for w in word:
            if w in data:
                data = data[w]
            else:
                return False
        if 'end' in data:
            return True
        return False

    def startsWith(self, prefix: str) -> bool:
        data = self.data
        for p in prefix:
            if p in data:
                data = data[p]
            else:
                return False
        return True

'''
212. Word Search II 单词搜索II
feeling: 写了前缀树和普通的DFS 都是超时 我感到心很累
'''
# 看完官方题解写的
class Solution:
    dires = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        res = set()
        m, n = len(board), len(board[0])
        datatire = {}
        for word in words:
            data = datatire
            for ch in word:
                # data = data.setdefault(ch, {})
                if ch not in data:
                    data[ch] = {}
                data = data[ch]
            data['end'] = {1}

        def search(i, j, restemp, data, marked):
            if 'end' in data:
                res.add(restemp)
            for dire in self.dires:
                new_i, new_j = i + dire[0], j + dire[1]
                if 0 <= new_i < m and 0 <= new_j < n and board[new_i][new_j] in data and (new_i, new_j) not in marked:
                    search(new_i, new_j, restemp + board[new_i][new_j], data[board[new_i][new_j]],
                           marked | {(new_i, new_j)})

        for i in range(m):
            for j in range(n):
                if board[i][j] in datatire:
                    marked = [[0 for _ in range(n)] for _ in range(m)]
                    search(i, j, board[i][j], datatire[board[i][j]], {(i, j)})
        return res

# 我的心酸超时代码之一：
from collections import defaultdict
import copy
class Solution:
    dires = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        datatire = defaultdict(dict)
        # 首先构造tire
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                self.dfs(i, j, m, n, board, {(i, j)}, datatire)

        res = []
        for word in words:
            data = datatire
            flag = 0
            for w in word:
                if w not in data:
                    flag = 1
                    break
                data = data[w]
            if flag == 0:
                res.append(word)
        return res

    def dfs(self, i, j, m, n, board, marked, datacop):
        data = datacop
        if board[i][j] not in data:
            data[board[i][j]] = {}
        data = data[board[i][j]]
        for dire in self.dires:
            new_i = i+dire[0]
            new_j = j+dire[1]
            if 0<=new_i<m and 0<=new_j<n and (new_i, new_j) not in marked:
                self.dfs(new_i, new_j, m, n, board, marked | {(new_i, new_j)}, data)

# 我的心酸超时代码之二：
from collections import defaultdict
class Solution:
    dires = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        res = []
        m, n = len(board), len(board[0])
        for word in words:
            for i in range(m):
                for j in range(n):
                    flag = 0
                    if self.search(i, j, m, n, word, board, {(i, j)}):
                        res.append(word)
                        flag = 1
                        break
                if flag == 1:
                    break
        return res

    def search(self, i, j, m, n, word, board, marked):
        # marked = copy.deepcopy(marked)
        # marked[i][j] = 1
        if word[0] != board[i][j]:
            return False
        if len(word) == 1:
            return True
        for dire in self.dires:
            new_i = i+dire[0]
            new_j = j+dire[1]
            if 0<=new_i<m and 0<=new_j<n and (new_i, new_j) not in marked:
                if not self.search(new_i, new_j, m, n, word[1:], board, marked | {(new_i, new_j)}):
                    continue
                else:
                    return True
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

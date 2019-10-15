# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Trie
   Description :
   Author :         amilyxy
   date：           2019/10/15
-------------------------------------------------
"""
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
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       String
   Description :
   Author :         amilyxy
   date：           2019/9/4
-------------------------------------------------
"""
'''
28. Implement strStr(): 实现 strStr()
describe: 给定一个 haystack 字符串和一个 needle 字符串，在haystack字符串中找出needle字符串出现的第一个位置(从0开始)。
          如果不存在，则返回-1。
'''
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)-len(needle)+1):
            if haystack[i: i+len(needle)] == needle:
                return i
        return -1

'''
14. Longest Common Prefix: 最长公共前缀
describe: 编写一个函数来查找字符串数组中的最长公共前缀。
          如果不存在公共前缀，返回空字符串 ""。
'''
class Solution:
    def longestCommonPrefix(self, strs: list[str]) -> str:
        # 方法1 44ms
        if len(strs) == 0:
            return ""
        L = min(map(len, strs))
        j = 0
        for i in range(L):
        # 其实这一步相当于zip的用法
            if len(set(map(lambda x: x[j], strs))) == 1:
                j += 1
            else:
                break
        return strs[0][:j]

        # 方法2
        strs.sort()
        if len(strs) == 0: return ""
        begin = strs[0]
        end = strs[-1]
        for i in range(len(begin)):
            if begin[i] != end[i]:
                return end[:i]
        return begin

'''
58. Length of Last Word: 最后一个单词的长度
describe: 给定一个仅包含大小写字母和空格' '的字符串，返回其最后一个单词的长度。
          如果不存在最后一个单词，请返回 0 。
'''
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        # 方法一
        if len(s) == 0: return 0
        s = s.split(' ')
        for i in range(len(s)-1,-1,-1):
            if s[i] != '':
                return len(s[i])
            if i == 0:
                return 0

        s = s.rstrip().split(' ')   #这里split(' ')带空格很重要
        return len(s[-1])

        #方法二 strip()真的解决了我写了一个小时“笨方法”出现的问题 就是不知道怎么定 begin和end 太难了鸭
        s = s.strip()
        begin = -1
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ' ':
                begin = i
                break
        if begin == -1:
            return len(s)
        else:
            return (len(s) - i - 1)

'''
387. First Unique Character in a String 字符串中的第一个唯一字符
'''
# 我的方法一
from collections import OrderedDict
class Solution:
    def firstUniqChar(self, s: str) -> int:
        sdict = OrderedDict()
        for i in s:
            sdict[i] = sdict.setdefault(i, 0) + 1
        sval = list(sdict.values())
        skey = list(sdict.keys())
        if 1 in sval:
            key = skey[sval.index(1)]
            print(key)
            return s.index(key)
        return -1

# 我的方法二 （这个时间真的短！！ 开心！！！
class Solution:
    def firstUniqChar(self, s: str) -> int:
        n = len(s)
        count = set()
        for i in range(n):
            if s[i] not in count:
                if s[i] not in s[i+1:]:
                    return i
                else:
                    count.add(s[i])
        return -1

'''
383. Ransom Note 赎金信
'''
# 这个方法先把ransom和magazine遍历一次 然后再判断 有点不太高效
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ransdict = {}
        magadict = {}
        for i in ransomNote:
            ransdict[i] = ransdict.setdefault(i, 0) + 1
        for i in magazine:
            magadict[i] = magadict.setdefault(i, 0) + 1
        for i, j in ransdict.items():
            if i not in magadict or j > magadict[i]:
                return False
        return True

# 我的天哪噜 我是不是做完一轮之后对时间特别敏感... 这个时间超级短！！
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        chars = set(ransomNote)
        for char in chars:
            i = ransomNote.count(char)
            j = magazine.count(char)
            if i >j:
                return False
        return True


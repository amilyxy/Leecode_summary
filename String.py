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
# 题解方法一 暴力滑动搜索
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)-len(needle)+1):
            if haystack[i: i+len(needle)] == needle:
                return i
        return -1

# 11.15 更新 sunday匹配方法（浩波来问的，不然我都不知道~
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # 计算偏移表
        dic = {}
        n = len(needle)
        n1 = len(haystack)
        for i in range(n - 1, -1, -1):
            if needle[i] not in dic:
                dic[needle[i]] = n - i
        dic['not'] = n + 1

        indx = 0
        while (indx + n) <= n1:
            temp = haystack[indx: indx + n]
            if temp == needle:
                return indx
            else:
                if (indx + n) < n1:
                    if haystack[indx + n] in dic:
                        indx += dic[haystack[indx + n]]
                    else:
                        indx += dic['not']
                else:
                    return -1
        return indx if (indx + n) < n1 else -1

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
        for skey, sval in sdict.items():
            if sval == 1:
                return s.index(skey)
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

# counter解法
class Solution:
    def firstUniqChar(self, s: str) -> int:
        from collections import Counter
        for k,v in Counter(s).items():
            if v==1:
                return s.index(k)
        return -1

# 题解方法  只需要遍历26次  @imckl
class Solution(object):
    def firstUniqChar(self, s: str) -> int:
        min_unique_char_index = len(s)
        for c in "abcdefghijklmnopqrstuvwxyz":
            i = s.find(c)
            if i != -1 and i == s.rfind(c):
                min_unique_char_index = min(min_unique_char_index, i)
        return min_unique_char_index if min_unique_char_index != len(s) else -1

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

# 一行解决
# collections.Counter(magazine) & collections.Counter(ransomNote) == collections.Counter(ransomNote)

'''
344. Reverse String 反转字符串
好像题目的规范解法是双指针
'''
class Solution:
    def reverseString(self, s: list[str]) -> None:
        # 方法一
        # s.reverse()

        # 方法二 交换
        n = len(s)-1
        for i in range(n//2+1):
            s[i], s[n-i] = s[n-i], s[i]

        #方法三 不知道这个算不算原地操作
        s[:] = s[::-1]

'''
151. Reverse Words in a String 反转字符串里的单词
'''
class Solution:
    # 方法一
    def reverseWords(self, s: str) -> str:
        temp = s.split()
        res = ""
        for i in temp[::-1]:
            res = res+i+' '
        res = res.rstrip()
        return res

    # 方法二
    def reverseWords(self, s: str) -> str:
        temp = s.split()
        return " ".join(temp[::-1])

    # 感觉还是要用原始方法会比较好 万一面试官说让你自己实现不用高级函数 GG
    def reverseWords(self, s: str) -> str:
        lists = []
        wordl=0
        wordr=0
        n = len(s)
        res = ""
        while wordr<n:
            while wordl<n and s[wordl]==' ':
                wordl+=1
            # 说明找到单词左边了
            wordr = wordl
            while wordr<n and s[wordr]!=' ':
                wordr +=1
            # 说明找到单词右边了
            if wordl<n:
                lists.append(s[wordl: wordr])
                wordl = wordr
        print(lists)
        for i in lists[::-1]:
            res = res+i+' '
        res = res.rstrip()
        return res

'''
459.重复的子字符串
'''
# 方法一 暴力重复
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        for i in range(1, n):
            if n%i == 0:
                if s[:i]*(n//i) == s:
                    return True
        return False


# kmp方法
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        pnext = [0, 0]
        j = 0
        for i in range(1, len(s)):
            while j > 0 and s[i] != s[j]:
                j = pnext[j]
            if s[j] == s[i]:
                j += 1
            pnext.append(j)
        return len(s) % (len(s)-pnext[-1]) == 0 and pnext[-1] > 0








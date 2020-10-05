# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       bit
   Description :
   Author :         amilyxy
   date：           2019/10/5
-------------------------------------------------
"""
'''
389. Find the Difference: 找不同
describe: 给定两个字符串 s 和 t，它们只包含小写字母。
          字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
          请找出在 t 中被添加的字母。
'''
import operator
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        # 方法一：逐个删除
        # emm这个不太好，i in list时间复杂度高也就算了，还把s/t中的元素修改了
        t = list(t)
        for i in s:
            t.remove(i)
        res = "".join(t)
        return res
        # replace做不需要转换成list
        # for i in s:
        #     t = t.replace(i, '', 1)

    # 方法二 按位比较，就是str不能直接排序有点烦
    def findTheDifference(self, s: str, t: str) -> str:
        s = list(s)
        s.sort()
        s = "".join(s)
        t = list(t)
        t.sort()
        t = "".join(t)
        i = 0
        while i<len(s):
            if operator.ne(s[i], t[i]):
                return t[i]
            i += 1
        if i == len(s):
            return t[i]

    # 方法三：ASCII之差
    def findTheDifference(self, s: str, t: str) -> str:
        res = chr(sum(map(ord, t)), sum(map(ord, s)))
        return res

    # 方法四： 异或法
    '''
    ⭐ 加精!  
    感觉这个才是题目所要求的的呀！
    '''
    def findTheDifference(self, s: str, t: str) -> str:
        res = 0
        for i in s:
            res ^= ord(i)
        for j in t:
            res ^= ord(j)
        return chr(res)

'''
136. Single Number: 找不同
describe: 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
'''
class Solution:
    # 这个时间复杂度为n^2 不好不好
    def singleNumber(self, nums: list[int]) -> int:
        nums.sort()
        for i in range(0, len(nums), 2):
            # nums[i] != nums[i+1]
            if i == (len(nums)-1) or operator.ne(nums[i], nums[i+1]):
                return nums[i]

    # 方法二：位操作
    '''
    ⭐ 加精!  
    感觉这个才是题目所要求的的呀！
    '''
    class Solution(object):
        def singleNumber(self, nums):
            res = 0
            for i in nums:
                res ^= i
            # reduce(lambda x, y: x^y, nums)
            return res

'''
318. Maximum Product of Word Lengths: 最大单词长度乘积
describe: 给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，
          并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。
          如果不存在这样的两个单词，返回 0。
'''
class Solution:
    # 题解方法一： @麦麦麦麦子。
    def maxProduct(self, words: list[str]) -> int:
        # 直观的版本，主要通过位运算来判断字母位
        max_len = {}
        for word in words:
            flag = 0  # flag用26位二进制表示该词使用了的字母
            for alp in word:
                flag |= 1 << ord(alp) - 97  # 置字母对应的二进制位为1
            max_len[flag] = max(max_len.get(flag, 0), len(word))  # 更新当前flag的最大长度
        # [0]用来避免对空列表取max，下面的比较次数为n^2
        return max([0] + [max_len[x] * max_len[y] for x in max_len for y in max_len if x & y == 0])

'''
201.数字范围按位与
'''
# 移位操作
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        i = 0
        while m != n:
            m >>= 1
            n >>= 1
            i += 1
        return m << i

# 方法二：
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        while n > m:  # 直到m大于等于n
            n &= (n-1)
        return n

'''
89.格雷编码  
'''
# 根据格雷码的生成公式
class Solution:
    def grayCode(self, n: int) -> List[int]:
        n = pow(2,n)
        res = []
        for i in range(n):
            # res.append(i^(int(i/2)))
            res.append(i^(i>>1))
        return res

# 根据格雷码的镜像排列规则 @jyd
class Solution:
    def grayCode(self, n: int) -> List[int]:
        res, head = [0], 1
        for i in range(n):
            for j in range(len(res) - 1, -1, -1):
                res.append(head + res[j])
            head <<= 1
        return res

# 镜像排列的另一种实现 @powcai
# 都是大佬哈！我的位运算还不够熟练
class Solution:
    def grayCode(self, n: int) -> List[int]:
        res = [0]
        for i in range(n):
            for j in range(len(res) - 1, -1, -1):
                res.append(res[j] ^ (1 << i))
        return res


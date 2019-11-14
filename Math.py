# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Math
   Description :
   Author :         amilyxy
   date：           2019/9/4
-------------------------------------------------
"""
'''
7. Reverse Integer: 整数反转
describe: 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
'''
class Solution:
    def reverse(self, x: int) -> int:
        # # 方法一 56ms
        neg = 0
        if x < 0:
            neg = 1
            x = abs(x)
        strx = str(x)
        out = 0
        for i in range(len(strx)):
            out += pow(10, i)*int(strx[i])
        # 说明x是正数
        if (neg == 0) & (out < (pow(2, 31)-1)):
            return out
        if (neg == 1) & (out < pow(2, 31)):
            return -out
        return 0

        # 方法二
        neg = 0
        out = 0
        if x < 0:
            neg = 1
            x = abs(x)
        while True:
            out = out * 10 + x % 10
            if x // 10 == 0:
                break
            x = x // 10
        if (neg == 0) & (out < (pow(2, 31) - 1)):
            return out
        if (neg == 1) & (out < pow(2, 31)):
            return -out
        return 0

'''
165. Compare Version Numbers: 比较版本号
describe: 比较两个版本号 version1 和 version2。
          如果 version1 > version2 返回 1
          如果 version1 < version2 返回 -1
          除此之外返回 0。
'''
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        list1 = version1.split('.')
        list2 = version2.split('.')
        L1 = len(list1)
        L2 = len(list2)
        L = max(L1, L2)
        if L1 < L2:
            list1.extend(['0'] * (L2 - L1))
        elif L1 > L2:
            list2.extend(['0'] * (L1 - L2))
        # 评论方法 -d很关键
        # d = L1 - L2
        # list1, list2 = list1 + [0] * d, list2 + [0] * -d
        for i in range(L):
            if (int(list1[i]) > int(list2[i])):
                return 1
            elif (int(list1[i]) < int(list2[i])):
                return -1
        return 0

'''
66. Plus One: 加一
describe: 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
          最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
          你可以假设除了整数 0 之外，这个整数不会以零开头。
'''
class Solution:
    def plusOne(self, digits: list[int]) -> list[int]:
        # 方法一 56ms
        L = len(digits)
        out = []
        num = 0
        for i in range(L):
            num += pow(10, i)*digits[L-1-i]
        num += 1
        while True:
            out.append(num%10)
            num = num//10
            if num == 0:
                return out[::-1]

        # 方法二 52ms
        L = len(digits)
        for i in range(L - 1, -1, -1):
            temp = digits[i] + 1
            digits[i] = temp % 10
            if temp >= 10:
                if i == 0:
                    digits.insert(0, temp // 10)
                    break
            else:
                break
        return digits

'''
8. String to Integer (atoi) 字符串转换整数
'''
# 感觉真的是笨方法hhhh
class Solution:
    def myAtoi(self, str: str) -> int:
        res, i, neg = 0, 0, 0
        strin = str.lstrip()
        n = len(strin)
        nums = set(['0','1','2','3','4','5','6','7','8','9'])
        if n == 0:
            return res
        if strin[0] == '-' or strin[0] == '+':
            i = 1
            if strin[0] == '-':
                neg = 1
        while i<n and strin[i] in nums:
            res = res*10 + int(strin[i])
            i+=1
        if neg:
            res = -res
        if res>pow(2,31)-1:
            res = pow(2,31)-1
        if res<-pow(2, 31):
            res = -pow(2, 31)
        return res

# 看到题目的第一眼我就觉得可以用正则，但是我不会用py的正则，答案肯定有人用呀
import re
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)

'''
258. Add Digits 各位相加
好吧，看答案大家好像都是用%9去做的 == 是我naive
'''
class Solution:
    def addDigits(self, num: int) -> int:
        res = 0
        while num:
            res += num%10
            num //=10
            if not num and res>=10:
                num = res
                res = 0
        return res
# 方法二： 巧了嘿！ 昨天正好学习了eval() 今天就能用上
    def addDigits(self, num: int) -> int:
        while len(str(num))>1:
            num = eval('+'.join(list(str(num))))
        return num
# 题解方法三  %9
    def addDigits(self, num: int) -> int:
        if num > 9:
            num = num % 9
            if num == 0:
                return 9
        return num

'''
67. Add Binary 二进制求和
题意还是按原始的方法做吧！ 感觉方法还是蛮多的比如异或 不写了~
'''
class Solution:
# 真的是最原始的加减法hhh
    def addBinary(self, a: str, b: str) -> str:
        na = len(a)
        nb = len(b)
        if na > nb:
            b = '0'*(na-nb) + b
        else:
            a = '0'*(nb-na) + a
        add = 0
        n =len(a)
        res = ''
        for i in range(n-1, -1, -1):
            temp = int(a[i]) + int(b[i]) + add
            res = str(temp%2) + res
            add = 0 if temp<2 else 1
        if add == 1:
            res = '1' + res
        return res
# 方法二： python的骚操作
    def addBinary(self, a: str, b: str) -> str:
        res = bin(int(a, 2) + int(b, 2))
        return res[2:]

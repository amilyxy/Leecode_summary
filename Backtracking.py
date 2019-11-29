# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Backtracking
   Description :
   Author :         amilyxy
   date：           2019/9/6
-------------------------------------------------
"""
'''
78. Subsets: 子集
describe: 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
'''
import itertools
# 今天的好像都是评论方法  雨我无瓜
class Solution:
    #  方法一 库函数法
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = []
        for i in range(len(nums)+1):
            for tmp in itertools.combinations(nums, i):
                # 注意combinations得到的是一个对象 而tmp是一个tuple
                res.append(tmp)
        return res

    # 方法二 迭代
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
            # res.extend([[i] + z for z in res])
        return res

    # 方法三 递归（回溯）算法1
    '''
    ⭐ 加精!
    '''
    def subsets(self, nums: list[int]) -> list[list[int]]:
        res = []
        n = len(nums)
        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, n):
                helper(j + 1, tmp + [nums[j]])
        helper(0, [])
        return res

    # 方法四： 回溯算法2 + stack
    '''
    ⭐ 加精!  
    配合LC吐血整理的图，类似于树的遍历
    st 记录遍历的顺序
    '''
    def subsets(self, nums):
        L = len(nums)
        if L == 0:
            return []
        res = []
        def helper(start, st):
            # 真的是很奇怪，这里不能用res.append(st) 真的是搞不懂  希望有人解释一下
            res.append(st[:])
            for i in range(start, L):
                st.append(nums[i])
                helper(i+1, st)
                st.pop()
        helper(0, [])
        return res

    # 方法五： 回溯算法2 + stack
    '''
    ⭐ 加精!  
    根据子集的长度从[0, len(nums)]进行遍历回溯, 其实和前面的差不多
    '''
    def subsets(self, nums):
        L = len(nums)
        res = []
        if L == 0:
            return []
        # 遍历深度从0~len(nums)
        def helper(depth, start, st):
            if len(st) == depth:
                res.append(st[:])
                return
            for i in range(start, L):
                st.append(nums[i])
                helper(depth, i+1, st)
                st.pop()
        for i in range(L+1):
            helper(i, 0, [])
        return res

    # 方法六：二进制掩码的方法
    def subsets(self, nums: list[int]) -> list[list[int]]:
        size = len(nums)
        n = 1 << size
        res = []
        for i in range(n):
            cur = []
            for j in range(size):
                if i >> j & 1:
                    cur.append(nums[j])
            res.append(cur)
        return res

'''
90. SubsetsII: 子集II
describe: 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）
          解集不能包含重复的子集。
'''
class Solution:
    # 方法一 先找出所有子集 在逐个判断是否重复
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        res = [[]]
        out = []
        for i in nums:
            res += [[i] + z for z in res]
        for j in res:
            if sorted(j) in out:
                continue
            else:
                out.append(sorted(j))
        return out

    # 以下为题解方法
    # 方法二：trick:根据nums中数字的频数，不用去重
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        # 构造字典
        dic = {}
        for i in nums:
            dic[i] = dic.get(i, 0) +1
        for key, val in dic.items():
            temp = res.copy()
            for j in res:
                temp.extend(j+[key]*(k+1) for k in range(val))
            res = temp
        return res

'''
77. Combinations: 组合
describe: 给定两个整数 n 和 k，返回 1 ... n 中所有可能的k个数的组合。
'''
class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:
        # 方法1 回溯法(找树的宽度)
        if n == 0 or k > n:
            return []
        else:
            nums = [i for i in range(1, n+1)]
        res = []
        def findcombine(start, st):
            if len(st) == k:
                res.append(st[:])
                return
            for i in range(start, n):
                st.append(nums[i])
                findcombine(i+1, st)
                st.pop()
        findcombine(0, [])
        return res

    def combine(self, n: int, k: int) -> list[list[int]]:
        # 方法2 列举+筛选
        if n == 0 or k > n:
            return []
        else:
            nums = [i for i in range(1, n+1)]
        res = []
        subset = [[]]
        for i in nums:
            subset += [[i] + j for j in subset]
        for z in subset:
            if len(z) == k:
                res.append(z)
        # subset = list(filter(lambda x: len(x) == k, subset))
        return res

    # 方法三  用模块方法
    def combine(self, n: int, k: int) -> list[list[int]]:
        res = [i for i in itertools.combinations(range(1, n+1), k)]
        return res

    # 好像还可以用掩码的方法
'''
39. Combination Sum 组合总和
'''
# z只是想把代码发上来看看我有多奇葩 在超时的边缘反复试探
class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        res1 = []
        # 从结果出发：
        def helper(t, res):
            if sum(res) == target:
                flag = 0
                for i in res1:
                    if len(i) == len(res) and set(i) == set(res) and Counter(i) == Counter(res):
                        flag = 1
                        break
                if not flag:
                    res1.append(res)
            # 以上if可替代方法 万事皆有解决方法嘛！
            '''
            if sum(temp) == target and sorted(temp) not in res:
                res.append(sorted(temp))           
            '''
            if sum(res) < target:
                for i in range(1, t + 1):
                    if i in candidates:
                        helper(t - i, res+[i])

        helper(target, [])
        return res1

# 题解方法@powcai 还是觉得有点难理解,写不出来也是真的
class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates.sort()
        n = len(candidates)
        res = []
        def backtrack(i, tmp_sum, tmp):
            if  tmp_sum > target or i == n:
                return
            if tmp_sum == target:
                res.append(tmp)
                return
            for j in range(i, n):
                if tmp_sum + candidates[j] > target:
                    break
                backtrack(j,tmp_sum + candidates[j],tmp+[candidates[j]])
        backtrack(0, 0, [])
        return res

# 动态规划方法 @蠢萌哒小洋  我真的跪了...
class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        dict = {}
        for i in range(1, target + 1):
            dict[i] = []

        for i in range(1, target + 1):
            for j in candidates:
                if i == j:
                    dict[i].append([i])
                elif i > j:
                    for k in dict[i - j]:
                        x = k[:]
                        x.append(j)
                        x.sort()  # 升序，便于后续去重
                        if x not in dict[i]:
                            dict[i].append(x)

        return dict[target]

'''
40. Combination Sum II 组合总和 II
'''
# 今天依旧是在超时边缘试探的一天 ==
from collections import Counter
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        sorted(candidates)
        res = []
        def helper(t, temp, cand):
            if sum(temp) == target:
                flag = 0
                for j in res:
                    if len(j) == len(temp) and set(j) == set(temp) and Counter(temp) == Counter(j):
                        flag = 1
                        break
                if not flag:
                    res.append(temp)
            if sum(temp) < target:
                for i in cand:
                    if i <= t:
                        cand1 = cand.copy()
                        cand1.remove(i)
                        helper(t-i, temp+[i], cand1)

        helper(target, [], candidates)
        return(res)




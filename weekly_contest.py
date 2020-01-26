# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       week_competition
   Description :
   Author :         amilyxy
   date：           2019/12/23
-------------------------------------------------
"""
'''
1295: 划分数组为连续数字集合
'''
# 老样子 第一个还是超时程序 在36个案例时超时
class Solution:
    def isPossibleDivide(self, nums: list[int], k: int) -> bool:
        nums.sort(reverse=True)
        while len(nums):
            count = 0
            last = nums[-1]-1
            n = len(nums)
            for i in range(n-1, -1, -1):
                print(nums[i])
                print(last)
                if count<=k-1:
                    if nums[i] != last:
                        if (nums[i]-last) != 1:
                            return False
                        last = nums[i]
                        count+=1
                        nums.pop(i)
                else:
                    break
            if count<=k-1:
                return False
        return True

# 哈哈哈哈做出来了！
from collections import Counter
class Solution:
    def isPossibleDivide(self, nums: list[int], k: int) -> bool:
        nums = Counter(nums)
        if len(nums) == k and len(set(nums.values())) == 1:
            return True
        else:
            while len(nums):
                cut = {}
                keys = set(nums)
                begin = min(keys)
                end = begin+k
                freq = nums[begin]
                for i in range(begin, end):
                    if i in keys and nums[i] >= freq:
                        cut[i] = freq
                    else:
                        return False
                nums = nums-Counter(cut)
            return True

'''
1305. 两棵二叉搜索树中的所有元素
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        self.res = []

        def helper(node):
            if node is None:
                return
            helper(node.left)
            self.res.append(node.val)
            helper(node.right)

        helper(root1)
        helper(root2)
        self.res.sort()
        return self.res

'''
1306. 跳跃游戏 III
'''
from collections import defaultdict
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        # 先构建一个下标图
        dictarr = defaultdict(set)
        n = len(arr)
        for i in range(n):
            val = arr[i]
            if i - val >= 0:
                dictarr[i].add((i - val))
            if i + val < n:
                dictarr[i].add((i + val))
                # print(dictarr)

        def dfs(a, marked):
            if arr[a] == 0:
                return True
            for i in dictarr[a]:
                if i not in marked:
                    marked.append(i)
                    if dfs(i, marked): return True

        marked = [start]
        if dfs(start, marked):
            return True
        else:
            return False

'''
1309. 解码字母到整数映射
'''
# 方法1 做题的时候想到的
class Solution:
    def freqAlphabets(self, s: str) -> str:
        flag = 0
        if s[-1] == '#': flag = 1
        s = s.split('#')
        if flag == 1: s = s[:-1]
        res = []
        lens = len(s)
        for i in range(lens):
            end = len(s[i])-2
            if i == (lens-1) and not flag:
                end = len(s[i])
            for j in s[i][:end]:
                res.append(chr(int(j)+96))
            if s[i][end:]:
                res.append(chr(int(s[i][end:])+96))
        return ''.join(res)

# 后面想到的 算是双指针吧
class Solution:
    def freqAlphabets(self, s: str) -> str:
        res = []
        lens = len(s)
        i = 0
        while i<len(s):
            if i+2<lens and s[i+2] == '#':
                res.append(chr(int(s[i:i+2])+96))
                i += 2
            else:
                res.append(chr(int(s[i])+96))
            i += 1
        return ''.join(res)

'''
1310. 子数组异或查询
'''
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        res = []
        xorarr = [0, arr[0]]
        for i in arr[1:]:
            xorarr.append(xorarr[-1] ^ i)
        for i in queries:
            res.append(xorarr[i[0]] ^ xorarr[i[1] + 1])
        return res

'''
1311. 获取你好友已观看的视频
'''
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        lev = 0
        has_focus, now_focus = {id}, {id}
        while lev<level:
            temp = []
            for i in now_focus:
                temp.extend(friends[i])
            now_focus = set(temp)
            if lev != level-1:
                has_focus = has_focus|now_focus
            else:
                now_focus = now_focus - has_focus
            lev += 1
        dic = {}
        for i in now_focus:
            for j in watchedVideos[i]:
                dic[j] = dic.setdefault(j, 0)+1
        dic = sorted(dic.items(), key=lambda x:(x[1], x[0]))
        res = [i[0] for i in dic]
        return res

# 第171次周赛
'''
5307. 将整数转换为两个无零整数的和
'''
# 当然也可以通过 %10?=0 来判断是否有0，我觉得我是投机取巧了..
class Solution:
    def getNoZeroIntegers(self, n: int) -> List[int]:
        for i in range(1, (n + 1) // 2 + 1):
            if '0' not in str(i):
                j = n - i
                if '0' not in str(j):
                    return [i, j]


'''
5308. 或运算的最小翻转次数
'''
import itertools
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        a = bin(a)[2:][::-1]
        b = bin(b)[2:][::-1]
        c = bin(c)[2:][::-1]
        temp = list(itertools.zip_longest(a, b, c, fillvalue='0'))
        # print(list(temp))
        res = 0
        for i in temp:
            if i[2] == '1' and '1' not in i[:2]:
                res += 1
            if i[2] == '0' and '1' in i[:2]:
                if '0' in i[:2]:
                    res += 1
                else:
                    res += 2
        return res

'''
5309. 连通网络的操作次数
'''
# DFS方法  564ms
from collections import defaultdict
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        # 线缆总数
        num = len(connections)
        # 构建图
        graph = defaultdict(list)
        for i in connections:
            graph[i[0]].append(i[1])
            graph[i[1]].append(i[0])

        marked = set()
        # 求连通块数
        cnum = 0
        for i in graph.keys():
            if i not in marked:
                self.dfs(marked, i, graph)
                cnum += 1

        if len(connections) - (len(graph) - 1) < (n - len(marked)):
            return -1
        else:
            return cnum - 1 + (n - len(marked))

    def dfs(self, marked, node, graph):
        marked.add(node)
        for j in graph[node]:
            if j not in marked:
                self.dfs(marked, j, graph)

# Quick-find 方法在n=10000 第32个测试用例超时
from collections import defaultdict
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
       # 设置查集
        dic = {}
        for i in range(n):
            dic[i] = i
        for i in connections:
            if dic[i[0]]!=dic[i[1]]:
                temp = dic[i[0]]
                for j in range(n):
                    if dic[j] == temp:
                        dic[j] = dic[i[1]]
        print(dic)
        if len(connections)< n-1:
            return -1
        else:
            return len(set(dic.values()))-1

# Quick-union 方法1252ms
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        tree = [i for i in range(n)]
        count = n

        def findroot(node):
            while node != tree[node]:
                node = tree[node]
            return node

        for i in connections:
            rootl = findroot(i[0])
            rootr = findroot(i[1])
            if rootl != rootr:
                tree[rootl] = rootr
                count -= 1

        if len(connections) < n - 1:
            return -1
        else:
            return count - 1

# weight quick union 988ms  感觉也没比quick-union快啊...
from collections import defaultdict
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        tree = [i for i in range(n)]
        size = [1 for _ in range(n)]
        count = n
        def findroot(node):
            while node != tree[node]:
                node = tree[node]
            return node

        for i in connections:
            rootl = findroot(i[0])
            rootr = findroot(i[1])
            if rootl != rootr:
                if size[rootl]<size[rootr]:
                    tree[rootl] = rootr
                    size[rootl]+=size[rootr]
                else:
                    tree[rootr] = rootl
                    size[rootr] += size[rootl]
                count -= 1
        # print(tree)
        if len(connections)< n-1:
            return -1
        else:
            return count-1

# @tuotuoli  大神的做法，膜拜一下
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1:
            return -1
        p = [[i] for i in range(n)]
        for x, y in connections:
            if p[x] is not p[y]:
                if len(p[x]) < len(p[y]):
                    x, y = y, x
                p[x].extend(p[y])
                for z in p[y]:
                    p[z] = p[x]
        return len({*map(id, p)}) - 1

'''
5315. 6和9组成的最大数字
'''
# 纯数学解决方法
class Solution:
    def maximum69Number(self, num: int) -> int:
        n = len(str(num))
        res = 0
        for i in range(n, -1, -1):
            temp = num // (10 ** i)
            num = num % (10 ** i)
            if temp == 6:
                res = res * 10 + 9
                res = res * (10 ** i) + num
                break
            else:
                res = res * 10 + temp
        return res

# str+内置函数：
class Solution:
    def maximum69Number (self, num: int) -> int:
        # 将str中
        return int(str(num).replace('6','9',1))

# 纯str：
class Solution:
    def maximum69Number (self, num: int) -> int:
        num = list(str(num))
        for i in range(len(num)):
            if num[i] == '6':
                num[i] = '9'
                break
        return ''.join(num)

'''
5316.竖直打印的单词
'''
class Solution:
    def printVertically(self, s: str) -> List[str]:
        s_zipped = list(itertools.zip_longest(*s.split(' '), fillvalue=' '))
        return [''.join(i).rstrip() for i in s_zipped]

'''
5317.删除给定的叶子节点
'''
# 我还是那个我。写代码最长的我。最耗时的我。
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        self.flag = 1
        new = TreeNode('#')
        new.left = root
        print(new)

        def helper(father, node, dire):
            if node.left == None and node.right == None and node.val == target:
                self.flag = 1
                if dire == 'left':
                    father.left = None
                if dire == 'right':
                    father.right = None
            if node.left != None:
                helper(node, node.left, 'left')
            if node.right != None:
                helper(node, node.right, 'right')

        while self.flag:
            self.flag = 0
            if new.left != None:
                helper(new, new.left, 'left')
            else:
                break

        return new.left

# 土旺大佬的做法 单次遍历就好
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        def helper(node):
            if not node:
                return True
            if helper(node.left):
                node.left = None
            if helper(node.right):
                node.right = None
            if not node.left and not node.right and node.val==target:
                return True
            else:
                return False

        temp = helper(root)
        if temp:
            return None
        else:
            return root

# 土旺大佬修改之后的  想想还是觉得自己写的好辣鸡啊
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        def dfs(node):
            if not node:
                return None
            node.left = dfs(node.left)
            node.right = dfs(node.right)
            if not node.left and not node.right and node.val == target:
                return None
            else:
                return node
        return dfs(root)

###173场周赛
'''
5319.删除回文子序列
'''
#  我还是不懂怎么做？？？莫名其妙
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        if s=="":
            return 0
        if s==s[::-1]:
            return 1
        else:
            return 2

'''
5320.餐厅过滤器 
'''
# 方法一 最开始写的
class Solution:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
        rest_unzip = list(zip(*restaurants))
        m = len(rest_unzip[0])
        res = [0 for _ in range(m)]
        match = [veganFriendly, maxPrice, maxDistance]
        for i in range(2, 5):
            if i == 2:
                for j in range(m):
                    if rest_unzip[i][j] >= match[i-2]:
                        res[j] = 1
            else:
                for j in range(m):
                    if res[j] == 1 and rest_unzip[i][j] > match[i-2]:
                        res[j] = 0
        restaurants_1 = [restaurants[i] for i in range(len(res)) if res[i] == 1]
        restaurants_1 = sorted(restaurants_1, key = lambda x:(x[1], x[0]), reverse=True)
        return [i[0] for i in restaurants_1]

# 简化方法
class Solution:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
        res = []
        for i in restaurants:
            if i[2] >= veganFriendly and i[3]<=maxPrice and i[4]<=maxDistance:
                res.append(i)
        res.sort(key=lambda x: (-x[1], -x[0]))
        return [i[0] for i in res]












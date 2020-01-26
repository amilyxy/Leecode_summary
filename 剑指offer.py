# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       剑指offer
   Description :
   Author :         amilyxy
   date：           2020/1/23
-------------------------------------------------
"""
'''
1.二维数组中的查找
'''
# 原始方案：暴力查找 时间复杂度O(n^2)
class Solution:
    def Find(self, target, array):
        m = len(array)
        n = len(array[0])
        for i in range(m):
            for j in range(n):
                if array[i][j] == target:
                    return True
        return False

# 优化方案一 ： 左上寻找 逐步减少查询范围   时间复杂度最多能O(n^2),好像并没有很优化多少
class Solution:
    def Find(self, target, array):
        m = len(array)
        n = len(array[0])
        temp = 0
        while temp<m and temp<n:
            for i in range(temp, m):
                if array[i][temp] == target:
                    return True
                if array[i][temp] > target:
                    m = i
                    break
            for j in range(temp, n):
                if array[temp][j] == target:
                    return True
                if array[temp][j] > target:
                    n = j
                    break
            temp+=1
        return False

# 优化方案二 右上查询 时间复杂度O(m+n)
class Solution:
    def Find(self, target, array):
        n = len(array[0])
        row, col = 0, n-1
        while row<n and col>-1:
            if array[row][col] == target:
                return True
            if array[row][col]<target:
                row+=1
            else:
                col-=1
        return False

'''
2. 替换空格
'''
class Solution:
    def replaceSpace(self, s):
        s = list(s)
        for i in range(len(s)-1, -1, -1):
            if s[i] == ' ':
                s[i] = '%20'
        return ''.join(s)

'''
3. 从尾到头打印链表
'''
# 方案一 顺序遍历逆序输出
class Solution:
    def printListFromTailToHead(self, listNode):
        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        return res[::-1]

# 方案二 回溯
class Solution:
    def printListFromTailToHead(self, listNode):
        # write code
        res = []
        def back(node):
            if node.next:
                back(node.next)
            res.append(node.val)
        if listNode:
            back(listNode)
        return res

'''
4. 重建二叉树
'''
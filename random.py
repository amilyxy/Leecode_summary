# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       random
   Description :
   Author :         amilyxy
   date：           2019/10/10
-------------------------------------------------
"""
'''
384. Shuffle an Array 打乱数组
'''
# 方法一：其实吧 也是调用random模块 产生随机数 选择index入numsrandom
import random
class Solution:
    def __init__(self, nums: list[int]):
        self.nums = nums
        # 工厂函数list(nums)也可以copy

    def reset(self) -> list[int]:
        return self.nums
    '''
    这多简单 直接调用函数 == 
    random.shuffle(self.numsrandom)       
    '''
    def shuffle(self) -> list[int]:
        data = list(self.nums)
        numsrandom = list(self.nums)
        for i in range(len(numsrandom)):
            remove = random.randrange(len(data))
            numsrandom[i] = data.pop(remove)
        return numsrandom

# 方法二：洗牌算法 就想小时候玩扑克的时候洗牌抽来抽去？？
class Solution:
    def __init__(self, nums: list[int]):
        self.nums = nums
        # 工厂函数list(nums)也可以copy

    def reset(self) -> list[int]:
        return self.nums
    '''
    这多简单 直接调用函数 == 
    random.shuffle(self.numsrandom)       
    '''
    def shuffle(self) -> list[int]:
        numsrandom = list(self.nums)
        for i in range(len(numsrandom)):
            swap = random.randrange(i, len(numsrandom))
            numsrandom[i], numsrandom[swap] = numsrandom[swap], numsrandom[i]
        return numsrandom

'''
384. Random Pick Index 随机数索引
'''
class Solution:
    def __init__(self, nums: list[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        # res = []
        # for ind, val in enumerate(self.nums):
        #     if val == target:
        #         res.append(ind)
        # random.shffle(res)
        # num = random.randint(0, len(res)-1)
        # return res[num]
        count = 0
        res = None
        for i, v in enumerate(self.nums):
            if v == target:
                if not random.randrange(0, count):
                    res = i
                count += 1
        return res

'''
382. Linked List Random Node 链表随机节点
'''
# 我有个疑惑 这个执行时间是不是有点长.....
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
import random
class Solution:
    def __init__(self, head: ListNode):
        self.head = head

    def getRandom(self) -> int:
        res = None
        count = 0
        node = self.head
        while node:
            count += 1
            if not random.randrange(count):
                res = node.val
            node = node.next
        return res

'''
380. 常数时间插入、删除和获取 Insert Delete GetRandom O(1)
'''
import random


class RandomizedSet:
    def __init__(self):
        self.li = set()

    def insert(self, val: int) -> bool:
        if val in self.li:
            return False
        self.li.add(val)
        return True

    def remove(self, val: int) -> bool:
        if val in self.li:
            self.li.remove(val)
            return True
        else:
            return False

    def getRandom(self) -> int:
        listli = list(self.li)
        i = random.randrange(len(self.li))
        return listli[i]









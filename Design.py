# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Design
   Description :
   Author :         amilyxy
   date：           2019/11/1
-------------------------------------------------
"""
'''
284. Peeking Iterator 顶端迭代器
'''
# 我差点都要脱口而出 copy大法好了....
import copy
class PeekingIterator:
    def __init__(self, iterator):
        self.cur = iterator

    def peek(self):
        temp = copy.deepcopy(self.cur)
        return temp.next()

    def next(self):
        if self.cur.hasNext():
            return self.cur.next()

    def hasNext(self):
        return self.cur.hasNext()

# 正确姿势使用迭代器
import copy


class PeekingIterator:
    def __init__(self, iterator):
        self.cur = iterator
        self.val = None

    def peek(self):
        # temp = copy.deepcopy(self.cur)
        if not self.val:
            self.val = self.cur.next()
        return self.val

    def next(self):
        # val超前一步
        if self.val:
            temp = self.val
            self.val = None
            return temp
        else:
            return self.cur.next()

    def hasNext(self):
        if self.val:
            return True
        return self.cur.hasNext()
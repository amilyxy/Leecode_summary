# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Stack&PriorityQueue
   Description :
   Author :         amilyxy
   date：           2019/10/2
-------------------------------------------------
"""
'''
155. Min Stack: 最小栈
describe: 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
'''
# 题解答案：@liweiwei1419
class MinStack:
    def __init__(self):
        # 数据栈
        self.data = []
        # 辅助栈
        self.helper = []

    def push(self, x):
        self.data.append(x)
        # 关键 1 和关键 2
        if len(self.helper) == 0 or x <= self.helper[-1]:
            self.helper.append(x)

    def pop(self):
        # 关键 3：【注意】不论怎么样，数据栈都要 pop 出元素
        top = self.data.pop()

        if self.helper and top == self.helper[-1]:
            self.helper.pop()
        return top

    def top(self):
        if self.data:
            return self.data[-1]

    def getMin(self):
        if self.helper:
            return self.helper[-1]

'''
232. Implement Queue using Stacks: 用栈实现队列
'''
# 看完官方题解后的python实现~
# 方法一，出队 入队分别在两个栈进行
# 用deque()模拟stack实现，只能使用append()和pop()
# 用deque()模拟stack实现，只能使用append()和pop()
from collections import deque
class MyQueue:
    # deque 模拟
    def __init__(self):
        self.st1 = deque()
        self.st2 = deque()

    def push(self, x: int) -> None:
        self.st1.append(x)

    def pop(self) -> int:
        if not self.st2:
            while self.st1:
                self.st2.append(self.st1.pop())
        return self.st2.pop()

    def peek(self) -> int:
        if not self.st2:
            while self.st1:
                self.st2.append(self.st1.pop())
        return self.st2[-1]

    def empty(self) -> bool:
        if self.st1 or self.st2:
            return False
        else:
            return True
# 方法二，用一个栈转存，出列 入列都在一个栈
class MyQueue:
    # deque 模拟
    def __init__(self):
        # 只能使用append()和pop()
        self.st1 = deque()
        self.st2 = deque()

    def push(self, x: int) -> None:
        self.st1.append(x)

    def pop(self) -> int:
        while self.st1:
            self.st2.append(self.st1.pop())
        temp = self.st2.pop()
        while self.st2:
            self.st1.append(self.st2.pop())
        return temp

    def peek(self) -> int:
        return self.st1[0]

    def empty(self) -> bool:
        if self.st1:
            return False
        else:
            return True

'''
252. Implement Stack using Queues: 用队列实现栈
'''
# 看完官方题解后的python实现~
# 方法一： 出列 入列全都在que1，que2用于暂存出队列的元素
# 用deque()模拟stack实现，只能使用append()和popleft()
class MyStack:
    def __init__(self):
        self.que1 = deque()
        self.que2 = deque()

    def push(self, x: int) -> None:
        self.que1.append(x)

    def pop(self) -> int:
        while len(self.que1)>1:
            self.que2.append(self.que1.popleft())
        out = self.que1.pop()
        self.que1, self.que2 = self.que2, self.que1
        return out

    def top(self) -> int:
        return self.que1[-1]

    def empty(self) -> bool:
        if self.que1:
            return False
        return True

# 方法二： 从形式上去模拟stack
# 用deque()模拟stack实现，只能使用append()和popleft()
from collections import deque
class MyStack:
    # deque()模拟
    def __init__(self):
        self.que1 = deque()
        self.que2 = deque()

    def push(self, x: int) -> None:
        while self.que1:
            self.que2.append(self.que1.popleft())
        self.que1.append(x)
        while self.que2:
            self.que1.append(self.que2.popleft())

    def pop(self) -> int:
        return self.que1.popleft()

    def top(self) -> int:
        return self.que1[0]

    def empty(self) -> bool:
        if self.que1:
            return False
        return True

# 方法三： 不需要两个queue，这个有点绕，需要理解
# 用deque()模拟stack实现，只能使用append()和popleft()
from collections import deque
class MyStack:
    # deque()模拟
    def __init__(self):
        self.que1 = deque()
        self.que2 = deque()

    def push(self, x: int) -> None:
        self.que1.append(x)
        size = len(self.que1)
        while size > 1:
            self.que1.append(self.que1.popleft())
            size -= 1

    def pop(self) -> int:
        return self.que1.popleft()

    def top(self) -> int:
        return self.que1[0]

    def empty(self) -> bool:
        if self.que1:
            return False
        return True






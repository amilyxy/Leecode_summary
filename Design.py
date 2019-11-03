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

'''
146. LRU Cache LRU缓存机制
'''
# 写的非常不友好 我知道 ==|  我不知道python有orderdict 而且 删除可以真的非常复杂
from collections import deque
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.data = deque()
        self.key = set()

    def get(self, key: int) -> int:
        temp = None
        for i in self.data:
            if key in i:
                temp = {key: i[key]}
                break
        if temp:
            self.data.remove(temp)
            self.data.append(temp)
            return temp[key]
        return -1

    def put(self, key: int, value: int) -> None:
        length = len(self.data)
        indata = {key: value}
        if key in self.key:
            for i in self.data:
                if key in i:
                    i[key] = value
                    temp = {key: value}
                    break
            self.data.remove(temp)
            self.data.append(temp)
        else:
            if length >= self.cap:
                a = self.data.popleft()
                for i in a:
                    self.key.remove(i)
            self.key.add(key)
            self.data.append(indata)

# 官方题解方法
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.data = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.data:
            self.data.move_to_end(key)
            self.data[key] = value
        else:
            if len(self.data) >= self.cap:
                self.data.popitem(last=False)
            self.data[key] = value

# 还有一个原始实现方法 感觉写的很不错 @Liye
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = {}
        # 新建两个节点 head 和 tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表为 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head

    # 因为get与put操作都可能需要将双向链表中的某个节点移到末尾，所以定义一个方法
    def move_node_to_tail(self, key):
            node = self.hashmap[key]
            node.prev.next = node.next
            node.next.prev = node.prev
            node.prev = self.tail.prev
            node.next = self.tail
            self.tail.prev.next = node
            self.tail.prev = node

    def get(self, key: int) -> int:
        if key in self.hashmap:
            self.move_node_to_tail(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value

    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            self.hashmap[key].value = value
            self.move_node_to_tail(key)
        else:
            if len(self.hashmap) == self.capacity:
                self.hashmap.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            # 如果不在的话就插入到尾节点前
            new = ListNode(key, value)
            self.hashmap[key] = new
            new.prev = self.tail.prev
            new.next = self.tail
            self.tail.prev.next = new
            self.tail.prev = new

'''
355. Design Twitter：设计推特
'''
from collections import defaultdict
from collections import OrderedDict

class Twitter:
    def __init__(self):
        self.follower = defaultdict(set)
        self.tweet = OrderedDict()

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweet[tweetId] = userId
        self.tweet.move_to_end(tweetId, last=False)

    def getNewsFeed(self, userId: int) -> List[int]:
        gettweet = []
        follwer = self.follower[userId]
        follwer.add(userId)
        for i in self.tweet:
            if (self.tweet[i] in follwer):
                gettweet.append(i)
            if len(gettweet) >= 10:
                break
        return gettweet

    def follow(self, followerId: int, followeeId: int) -> None:
        if followeeId not in self.follower[followerId]:
            self.follower[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.follower[followerId]:
            self.follower[followerId].remove(followeeId)

# 评论方法 @kingerious
'''
其实我做的时候也有想过要不要加时间戳 
可行是可行 而且运行时间好像也比我的少 
但是如果放在真实情况下 几亿用户发了十几亿甚至几百亿的推文 这样时间戳的方法是不是不太可行？？？？（瞎说的
'''
import collections
class Twitter:
    def __init__(self):
        self.dic = collections.defaultdict(list)
        self.timer = 0
        self.followDict = collections.defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.dic[userId].append([tweetId, self.timer])
        self.timer += 1

    def getNewsFeed(self, userId: int):
        News = []
        News += self.dic[userId]
        for follower in self.followDict[userId]:
            News += self.dic[follower]
        News = sorted(News, key=lambda x:x[1], reverse=True)
        return [news[0] for news in News][:10]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followeeId not in self.followDict[followerId] and followerId != followeeId:
            self.followDict[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.followDict[followerId]:
            self.followDict[followerId].remove(followeeId)


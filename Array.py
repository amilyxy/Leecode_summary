# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       Array
   Description :
   Author :         amilyxy
   date：           2019/9/4
-------------------------------------------------
"""
'''
27. Remove Element: 移除元素
describe: 给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
'''
class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        # 方法一
        nums[:] = list(filter(lambda x: x != val, nums))  # 加不加list其实都一样
        return len(nums)

        # 方法二
        L = len(nums)
        for i in range(L):
            try:
                nums.remove(val)
            except:
                break
        return len(nums)

        # 方法三
        j=len(nums)
        for i in range(j-1,-1,-1):
            if nums[i]==val:
                nums.pop(i)
        return len(nums)

'''
26. Remove Duplicates from Sorted Array: 删除排序数组中的重复项
describe: 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
'''
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # 方法一：时间太长 2400ms+
        a = list(set(nums))
        a.sort(key = nums.index)
        j = 0
        for i in a:
            nums[j] = i
            j += 1
        return j

        # 方法二：
        a = set(nums)
        j = 0
        for i in range(len(nums)):
            if nums[i] in a:
                nums[j] = nums[i]
                a.remove(nums[i])
                j += 1
                # 判断a是否为空
                if a.isEmpty():
                    break
        return j

        # # 这个很好啊 为啥会超出时间限制
        # j = 0
        # for i in range(len(nums)):
        #     if nums[i] not in nums[:i]:
        #         nums[j] = nums[i]
        #         j += 1
        # return j

        j = 1
        if len(nums) == 0:
            return 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[j] = nums[i]
                j += 1
        return j

'''
80. Remove Duplicates from Sorted Array II: 删除排序数组中的重复项II
describe: 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
'''
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # 方法一 80ms左右
        j = 1
        count = 0
        if len(nums) == 0: return 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                if count == 0:
                    count = 1
                    nums[j] = nums[i]
                    j += 1
            else:
                nums[j] = nums[i]
                count = 0
                j += 1
        return j

        # 方法二：时间好像不太行 感觉pop操作比赋值操作时间要长
        for i in range(len(nums)-1, 1, -1):
            if nums[i] == nums[i-2]:
                nums.pop(i)
        return len(nums)

        # 方法三：这个是本来有思路 但是以为不可行 看了下面那个又写了
        j = 2
        if len(nums) == 0: return 0
        if len(nums) == 1: return 1
        for i in range(2, len(nums)):
            if nums[i] != nums[j-2]:  # 不能i-2！！！ j-2 很关键！
                nums[j] = nums[i]
                j += 1
        return j

        # 评论解法
        i = 0
        for e in nums:
            if i < 2 or e != nums[i - 2]:
                nums[i] = e
                i += 1
        return i

'''
189. Rotate Array 旋转数组
注意是原地算法
'''
class Solution:
    def rotate(self, nums: list[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 方法一
        for i in range(k):
            nums.insert(0, nums.pop())

        # 方法二
        n = len(nums)
        k = k%n
        nums.extend(nums[:(n-k)])
        for i in nums[:(n-k)]:
            nums.remove(i)

        # 方法三
        n = len(nums)
        k = k%n
        for i in nums[:(n-k)]:
            nums.remove(i)
            nums.append(i)

        # 题解方法 @powcai
        nums = nums[::-1]
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]

'''
41. First Missing Positive 缺失的第一个正数
'''
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # 方法一：
        nums = set(nums)
        a = list(filter(lambda x: x>0, nums))
        a.sort()
        n = len(a)
        for i in range(1, n+2):
            if i==n+1 or i!=a[i-1]:
                return i

        # 方法二：
        a = set(filter(lambda x: x>0, nums))
        n = len(a)
        for i in range(1, n+2):
            if i not in a:
                return i

        # 方法三：
        if nums == []:
            return 1
        nums = set(nums)
        maxnum = max(nums)
        for i in range(1, maxnum + 2):
            if i not in nums:
                return i
        return 1

    # 题解方法@zhu_shi_fu
    def firstMissingPositive(self, nums: List[int]) -> int:
        if (not nums):
            return 1
        n = len(nums)
        for i in range(n):
            while (0 < nums[i] <= n and nums[i] != nums[nums[i] - 1]):
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        for i in range(n):
            if (nums[i] != i + 1):
                return i + 1
        return n + 1

'''
 299. Bulls and Cows 猜数字游戏
'''
 # 我的方法
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        bulls = 0
        cows = 0
        nums = 0
        n = len(guess)
        secretdict = {}
        for i in secret:
            secretdict[i] = secretdict.setdefault(i, 0) + 1
        for i in range(n):
            if guess[i] in secretdict:
                if secretdict[guess[i]]>0:
                    nums+=1
                    secretdict[guess[i]] -= 1
            if guess[i] == secret[i]:
                bulls +=1
        cows = nums-bulls
        res = ("%sA%sB" % (bulls, cows))
        return res

# 题解方法
# 好像题解方法也大同小异  没啥好写了
'''
315.计算右侧小于当前元素的个数
逆序对的应用
'''
# 归并排序方法 这个时间好像短一点
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums)<=1:
            return nums
        mid = (len(nums)+1)//2
        sub1 = self.sortArray(nums[:mid])
        sub2 = self.sortArray(nums[mid:])
        l, r = 0, 0
        tmp = []
        while l<len(sub1) or r<len(sub2):
            if r == len(sub2) or l<len(sub1) and sub1[l]<=sub2[r]:
                tmp.append(sub1[l])
                l+=1
            else:
                tmp.append(sub2[r])
                r+=1
        return tmp

# 二叉搜索树的做法
class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.leftsum = 0
        # 记录重复数字个数
        self.dup = 0

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        def insert(node, val):
            sumnode = 0
            while node.val != val:
                if node.val > val:
                    if not node.left:
                        node.left = Node(val)
                    node.leftsum += 1
                    node = node.left
                else:
                    # 如果没有重复数字 可以将node.dup直接改为1
                    sumnode += node.leftsum+node.dup
                    if not node.right:
                        node.right = Node(val)
                    node = node.right
            node.dup+=1
            return sumnode+node.leftsum

        if not nums: return []
        res = [0]*len(nums)
        root = Node(nums[-1])
        for i in range(len(nums)-1, -1, -1):
            res[i] = insert(root, nums[i])
        return res

'''
169.多数元素（简单
'''
# 摩尔投票  复杂度  时间O(n) 空间O(1)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        cand = None
        for i in nums:
            if count == 0:
                cand = i
            count += (1 if i == cand else -1)
        return cand

# 暴力哈希 时间O(n)  空间O(n) 不详细说了
# 快排 这个需要保证一定存在多数元素
class Solution:
    def majorityElement(self, numbers: List[int]) -> int:
        numbers.sort()
        return numbers[len(numbers)//2]

'''
974.和可被K整除的子数组
思路: ①前缀和 ②同余定理
'''
class Solution:
    def subarraysDivByK(self, A, K):
        dic = {0: 1}
        sumpre = 0
        for i in A:
            sumpre += i
            dic[sumpre % K] = dic.get(sumpre % K, 0) + 1

        return int(sum([dic[i] * (dic[i] - 1) / 2 for i in dic]))

'''
523.连续的子数组和
思路: 前缀和
'''
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        if len(nums) < 2: return False
        dp, cur = {0: -1}, 0
        for idx, num in enumerate(nums):
            cur += num
            if k != 0: cur %= k
            pre = dp.setdefault(cur, idx)
            if idx - pre > 1: return True
        return False

'''
88.合并两个有序数组
'''
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        while m>=0 and n:
            if m>0 and nums1[m-1]>nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1


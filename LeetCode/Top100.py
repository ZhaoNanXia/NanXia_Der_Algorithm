import collections
import random
from typing import List, Optional, Union

""" 哈希问题 """


def two_sum(nums, target):
    """
    LeetCode-1.两数之和
    时间复杂度：
    空间复杂度：
    """
    hash_table = collections.defaultdict(int)
    for i, num in enumerate(nums):
        if (target - num) in hash_table:
            return [i, hash_table[target - num]]
        else:
            hash_table[num] = i


# lists = [2, 7, 11, 15]
# print(two_sum(lists, 9))


def group_anagrams(strs):
    """
    LeetCode-49.字母异位词分组：给定一个字符串，将字母异位词组合在一起并返回结果列表
    时间复杂度：
    空间复杂度：
    """
    hash_table = collections.defaultdict(list)
    for word in strs:
        temp = ''.join(sorted(word))
        hash_table[temp].append(word)
    return list(hash_table.values())


# string = ["eat", "tea", "tan", "ate", "nat", "bat"]
# print(group_anagrams(string))


def longest_consecutive(nums):
    """
    LeetCode-128.最长连续序列：给定一个未排序数组，找出其中数字连续(不要求在原数组连续)的最长序列并返回长度
    """
    nums_set = set(nums)
    max_length = 0
    for num in nums_set:
        if (num - 1) not in nums_set:
            current_length = 0
            while num in nums_set:
                current_length += 1
                num += 1
            max_length = max(max_length, current_length)
    return max_length


# lists = [100, 4, 200, 1, 3, 2]
# print(longest_consecutive(lists))


""" 双指针问题 """


def move_zeroes(nums):
    """
    LeetCode-283.移动零：
    """
    i, j = 0, 0
    while j < len(nums):
        if nums[j] != 0:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
        j += 1
    return nums


# lists = [0, 1, 0, 3, 12]
# print(move_zeroes(lists))


def max_area(height):
    """
    LeetCode-11.盛最多水的容器：
    """
    left, right = 0, len(height) - 1
    area = 0
    while left < right:
        area = max(area, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return area


# lists = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(max_area(lists))


def three_sum(nums):
    """
    LeetCode-15.三数之和：
    """
    nums.sort()
    res = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left + 1] == nums[left]:
                    left += 1
                while left < right and nums[right - 1] == nums[right]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return res


# lists = [-1, 0, 1, 2, -1, -4]
# print(three_sum(lists))


def trap(height):
    """
    LeetCode-42.接雨水：
    """
    left, right = 0, len(height) - 1
    max_left, max_right = 0, 0
    res = 0
    while left < right:
        max_left, max_right = max(max_left, height[left]), max(max_right, height[right])
        if max_left < max_right:
            res += max_left - height[left]
            left += 1
        else:
            res += max_right - height[right]
            right -= 1
    return res


# lists = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
# print(trap(lists))


""" 滑动窗口问题 """


def length_of_longest_substring(s: str) -> int:
    """
    LeetCode-3.无重复字符的最长子串
    """
    windows = {}
    start = 0
    max_length = 0
    for i, c in enumerate(s):
        if c in windows and windows[c] >= start:
            start = windows[c] + 1
        else:
            max_length = max(max_length, i - start + 1)
        windows[c] = i
    return max_length


# string = "pwwkew"
# print(length_of_longest_substring(string))


def find_anagrams(s: str, p: str) -> List[int]:
    """
    LeetCode-438.找到字符串中所有字母异位词：
    """
    n_s, n_p = len(s), len(p)
    if n_s < n_p:
        return []
    need_p = collections.defaultdict(int)  # 记录p中字符的种类及数量
    for c in p:
        need_p[c] += 1
    window = collections.defaultdict(int)  # 统计当前窗口内的字符及数量
    valid = 0  # 记录当前窗口内满足need_p的字符种类数量
    left = 0  # 窗口的左边界
    res = []  # 记录结果
    for right, c in enumerate(s):
        if c in need_p:
            window[c] += 1
            if window[c] == need_p[c]:  # 若窗口内字符c的数量等于need_p中字符c的数量
                valid += 1  # 说明当前窗口已经满足need_p中字符种类的数量加一
        if right - left + 1 > n_p:  # 若窗口长度大于n_p，则要收缩左边界以维持窗口长度
            left_char = s[left]  # 左边界处的字符
            if left_char in need_p:  # 如果左边界处的字符也是目标字符之一
                if window[left_char] == need_p[left_char]:
                    valid -= 1
                window[left_char] -= 1
            left += 1
        # 如果字符种类满足条件且窗口长度匹配时记录结果
        if valid == len(need_p) and (right - left + 1) == len(p):
            res.append(left)
    return res


# string_s = "cbaebabacd"
# string_p = "abc"
# print(find_anagrams(string_s, string_p))


""" 子串问题 """


def subarray_sum(nums: List[int], k: int) -> int:
    """
    LeetCode-560.和为k的子数组
    前缀和
    """
    prefix_sum_frequency = collections.defaultdict(int)  # 记录前缀和出现的次数
    prefix_sum = 0  # 记录当前位置的前缀和
    prefix_sum_frequency[0] = 1  # 初始化前缀和0出现的次数为1
    res = 0
    for num in nums:
        prefix_sum += num
        # 已知当前位置i处的前缀和为prefix_sum，目标和为k，如果在这之前存在一个位置j的前缀和x，恰好有prefix_sum-x=k，
        # 那么区间[i,j]所囊括的元素组成的子数组和为k，即在存在前缀和x=prefix_sum-k时，更新和为k的子数组数量
        if prefix_sum_frequency[prefix_sum - k]:
            res += prefix_sum_frequency[prefix_sum - k]
        prefix_sum_frequency[prefix_sum] += 1
    return res


# lists = [1, 2, 3]
# print(subarray_sum(lists, 3))


def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """
    LeetCode-239.滑动窗口最大值：
    双端队列
    """
    deque = collections.deque()  # 使用双端队列存储窗口内元素下标且当前窗口的最大值下标始终位于队列头部
    res = []
    for i in range(len(nums)):
        # 若窗口右端位置为i，则左端位置应为i-k+1，故若窗口内元素下标小于i-k+1则要移除
        if deque and deque[0] < i - k + 1:
            deque.popleft()
        # 若当前元素小于队列末尾下标对应的元素则不管直接加入；
        # 若大于则剔除队列末尾下标往前查找，要保证队列中下标对应的元素是单调递减的才能使当前窗口的最大值下标始终位于队列头部
        while deque and nums[deque[-1]] < nums[i]:
            deque.pop()
        deque.append(i)
        # 从第一个窗口开始，记录最大值
        if i >= k - 1:
            res.append(nums[deque[0]])
    return res


# lists = [1, 3, -1, -3, 5, 3, 6, 7]
# print(max_sliding_window(lists, 3))


def min_window(s: str, t: str) -> str:
    """
    LeetCode-76.最小覆盖子串：
    """
    n_s, n_t = len(s), len(t)
    if n_s < n_t:
        return ''
    need_t = collections.defaultdict(int)
    count_t = 0
    for c in t:
        need_t[c] += 1
        count_t += 1
    start = 0
    res = (start, float('inf'))
    for i, c in enumerate(s):
        if need_t[c] > 0:
            count_t -= 1
        need_t[c] -= 1
        while count_t == 0:
            if i - start < res[1] - res[0]:
                res = (start, i)
            if need_t[s[start]] == 0:
                count_t += 1
            need_t[s[start]] += 1
            start += 1
    return '' if res[1] > n_s else s[res[0]: res[1] + 1]


# string_s = "ADOBECODEBANC"
# string_t = "ABC"
# print(min_window(string_s, string_t))


""" 普通数组问题 """


def max_subarray(nums):
    """
    LeetCode-53.最大子数组和：给定一个数组，找出一个具有最大和的连续子数组
    动态规划解法：求出以每个位置元素结尾的数组的最大连续子数组的和
    时间复杂度：O(n)，遍历数组一遍
    空间复杂度：O(1)
    """
    for i in range(1, len(nums)):
        # 以当前元素结尾的连续子数组的最大和取决于 ->
        # 以前一个元素结尾的连续子数组的最大和加上当前元素后的值 和 当前元素相比，取较大值
        nums[i] = max(nums[i], nums[i - 1] + nums[i])
    return max(nums)


# lists = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# print(max_subarray(lists))


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    LeetCode-56.合并区间：
    """
    intervals = sorted(intervals, key=lambda x: x[0])
    res = [intervals[0]]
    for interval in intervals:
        if interval[0] > res[-1][1]:
            res.append(interval)
        if interval[0] <= res[-1][1] <= interval[1]:
            res[-1][1] = interval[1]
    return res


# lists = [[1, 3], [2, 6], [8, 10], [15, 18]]
# print(merge(lists))


def rotate(nums: List[int], k: int):
    """
    LeetCode-189.轮转数组：
    """
    t = k % len(nums)  # 计算轮转结束后，最后几位数字将位于数组最前面
    nums[:] = nums[-t:] + nums[:-t]
    return nums


# lists = [1, 2, 3, 4, 5, 6, 7]
# print(rotate(lists, 3))


def product_except_self(nums: List[int]) -> List[int]:
    """
    LeetCode-238.除自身以外数组的乘积：
    """
    n = len(nums)
    res = [0] * n
    res[0] = 1
    suffix_product = 1  # 后缀积
    for i in range(1, n):
        res[i] = nums[i - 1] * res[i - 1]  # 计算前缀积
    for i in range(n - 2, -1, -1):
        suffix_product *= nums[i + 1]  # 计算后缀积
        res[i] *= suffix_product
    return res


# lists = [1, 2, 3, 4]
# print(product_except_self(lists))


def first_missing_positive(nums: List[int]) -> int:
    """
    LeetCode-41.缺失的第一个正数：
    将数组本身视为一个哈希表
    """
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


# lists = [3, 4, -1, 1]
# print(first_missing_positive(lists))

""" 矩阵问题 """


def set_zeroes(matrix: List[List[int]]) -> List[List[int]]:
    """
    LeetCode-73.矩阵置零
    """
    m, n = len(matrix), len(matrix[0])
    r, c = [False] * m, [False] * n
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                r[i] = True
                c[j] = True
    for i in range(m):
        for j in range(n):
            if r[i] or c[j]:
                matrix[i][j] = 0
    return matrix


# matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
# print(set_zeroes(matrix))


def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    LeetCode-54.螺旋矩阵
    """
    m, n = len(matrix), len(matrix[0])
    up, low, left, right = 0, m - 1, 0, n - 1
    res = []
    while True:
        for i in range(left, right + 1):
            res.append(matrix[up][i])
        up += 1
        if up > low:
            break
        for i in range(up, low + 1):
            res.append(matrix[i][right])
        right -= 1
        if right < left:
            break
        for i in range(right, left - 1, -1):
            res.append(matrix[low][i])
        low -= 1
        if up > low:
            break
        for i in range(low, up - 1, -1):
            res.append(matrix[i][left])
        left += 1
        if right < left:
            break
    return res


# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(spiral_order(matrix))


def rotate1(matrix: List[List[int]]) -> None:
    """
    LeetCode-48.旋转图像
    """
    m, n = len(matrix), len(matrix[0])
    # 矩阵转置，行列交换
    for i in range(m):
        for j in range(i, m):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 行反转
    for i in range(m):
        matrix[i] = matrix[i][::-1]


def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    LeetCode-240.搜索二维矩阵Ⅱ
    """
    def binary_search(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return False
    for i in range(len(matrix)):
        if binary_search(matrix[i], target):
            return True
    return False


def search_matrix_1(matrix: List[List[int]], target: int) -> bool:
    """
    LeetCode-240.搜索二维矩阵Ⅱ
    将矩阵逆时针旋转45°，可视为一个根节点为右上角元素的二叉搜索树
    """
    i, j = 0, len(matrix[0]) - 1
    while i < len(matrix) and j >= 0:
        if matrix[i][j] > target:
            j -= 1
        elif matrix[i][j] < target:
            i += 1
        else:
            return True
    return False


""" 链表问题 """


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    @staticmethod
    def array_to_linked_list(nums):
        """
        将一维数组转换成单向链表
        """
        if not nums:
            return None  # 空数组返回空链表
        dummy = ListNode(0)
        current = dummy
        for val in nums:
            current.next = ListNode(val)
            current = current.next
        return dummy.next

    @staticmethod
    def array_to_linked_lists_2d(nums):
        """
        将二维数组转换为链表列表
        """
        return [ListNode.array_to_linked_list(num) for num in nums]

    @staticmethod
    def linked_list_to_array(head):
        """
        将链表转换为数组
        """
        array = []
        current = head
        while current:
            array.append(current.val)  # 将当前节点的值添加到数组中
            current = current.next  # 移动到下一个节点
        return array


def get_intersection_node(head_a: ListNode, head_b: ListNode) -> Optional[ListNode]:
    """
    LeetCode-160.相交链表：
    """
    if not head_a or not head_b:
        return
    pa = head_a
    pb = head_b
    while pa != pb:
        pa = pa.next if pa else head_b
        pb = pb.next if pb else head_a
    return pa


# listA = [4, 1, 8, 4, 5]
# listB = [5, 6, 1, 8, 4, 5]
# headA = ListNode.array_to_linked_list(listA)
# headB = ListNode.array_to_linked_list(listB)
# print(get_intersection_node(headA, headB).val)


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-206.反转链表：
    """
    pre_node = None
    current_node = head
    while current_node:
        temp = current_node.next
        current_node.next = pre_node
        pre_node = current_node
        current_node = temp
    return pre_node


# lists = [1, 2, 3, 4, 5]
# lists_link = ListNode.array_to_linked_list(lists)
# print(ListNode.linked_list_to_array(reverse_list(lists_link)))


def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    LeetCode-234.回文链表：
    """
    if not head:
        return True
    node = []
    while head:
        node.append(head.val)
        head = head.next
    i, j = 0, len(node) - 1
    while i < j:
        if node[i] != node[j]:
            return False
        i += 1
        j -= 1
    return True


# lists = [1, 2, 2, 1]
# lists_link = ListNode.array_to_linked_list(lists)
# print(is_palindrome(lists_link))


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    LeetCode-141.环形链表：
    """
    if not head or not head.next:
        return False
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-142.环形链表Ⅱ：
    """
    if not head or not head.next:
        return None
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    if slow != fast:
        return None
    fast = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return fast if fast else None


def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-21.合并两个有序链表：
    """
    if not list1:
        return list2
    if not list2:
        return list1
    dummy_node = ListNode(0)
    current_node = dummy_node
    while list1 and list2:
        if list1.val < list2.val:
            current_node.next = list1
            list1 = list1.next
        else:
            current_node.next = list2
            list2 = list2.next
        current_node = current_node.next
    current_node.next = list1 if list1 else list2
    return dummy_node.next


# l1 = [1, 2, 4]
# l1_link = ListNode.array_to_linked_list(l1)
# l2 = [1, 3, 4]
# l2_link = ListNode.array_to_linked_list(l2)
# print(ListNode.linked_list_to_array(merge_two_lists(l1_link, l2_link)))


def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-2.两数相加：
    """
    if not l1:
        return l2
    if not l2:
        return l1
    carry = 0
    dummy_node = ListNode(0)
    current_node = dummy_node
    while l1 or l2 or carry:
        total_sum = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
        carry = total_sum // 10
        node = ListNode(total_sum % 10)
        current_node.next = node
        current_node = current_node.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy_node.next


# list1 = [2, 4, 3]
# l1_link = ListNode.array_to_linked_list(list1)
# list2 = [5, 6, 4]
# l2_link = ListNode.array_to_linked_list(list2)
# print(ListNode.linked_list_to_array(add_two_numbers(l1_link, l2_link)))


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    LeetCode-19.删除链表的倒数第N个结点：
    """
    if not head:
        return head
    fast = head
    dummy_node = ListNode(0)
    dummy_node.next = head
    slow = dummy_node
    for _ in range(n):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy_node.next


# list1 = [1, 2, 3, 4, 5]
# l1_link = ListNode.array_to_linked_list(list1)
# print(ListNode.linked_list_to_array(remove_nth_from_end(l1_link, 2)))


def swap_pairs(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-24.两两交换链表中的节点：
    """
    if not head or not head.next:
        return head
    dummy = ListNode(0)
    pre_node = dummy
    while head and head.next:
        first, second = head, head.next
        # 交换节点位置
        pre_node.next = second
        first.next = second.next
        second.next = first
        # 移动节点
        pre_node = first
        head = first.next
    return dummy.next


# list1 = [1, 2, 3, 4]
# l1_link = ListNode.array_to_linked_list(list1)
# print(ListNode.linked_list_to_array(swap_pairs(l1_link)))


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    LeetCode-25.k个一组翻转链表
    时间复杂度：O(n)
    空间复杂度：O(1)
    """
    def reverse_link(head_k, tail_k):
        if not head_k:
            return head_k
        pre_node_k = tail_k.next
        current_node = head
        while pre_node_k != tail:
            temp_k = current_node.next
            current_node.next = pre_node_k
            pre_node_k = current_node
            current_node = temp_k
        return tail, head

    dummy_node = ListNode(0)
    dummy_node.next = head
    pre_node = dummy_node
    while head:
        tail = pre_node
        for _ in range(k):
            tail = tail.next
            if not tail:
                return dummy_node.next
        temp = tail.next
        head, tail = reverse_link(head, tail)
        pre_node.next = head
        tail.next = temp
        pre_node = tail
        head = tail.next
    return dummy_node.next


# list1 = [1, 2, 3, 4, 5]
# l1_link = ListNode.array_to_linked_list(list1)
# print(ListNode.linked_list_to_array(reverse_k_group(l1_link, 3)))


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    LeetCode-148.排序链表：
    """
    nums_head = ListNode.linked_list_to_array(head)
    nums_head.sort()
    head = ListNode.array_to_linked_list(nums_head)
    return head


# list1 = [4, 2, 1, 3]
# l1_link = ListNode.array_to_linked_list(list1)
# print(ListNode.linked_list_to_array(sort_list(l1_link)))


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    LeetCode-23.合并k个升序链表：
    """
    if not lists:
        return
    dummy_node = ListNode(0)
    while len(lists) > 1:
        i, j = 0, len(lists) - 1
        while i < j:
            l1, l2 = lists[i], lists[j]
            current_node = dummy_node
            while l1 and l2:
                if l1.val < l2.val:
                    current_node.next = l1
                    l1 = l1.next
                else:
                    current_node.next = l2
                    l2 = l2.next
                current_node = current_node.next
            current_node.next = l1 if l1 else l2
            lists[i] = dummy_node.next
            lists.pop()
            i += 1
            j -= 1
    return lists[0]


# list1 = [[1, 4, 5], [1, 3, 4], [2, 6]]
# l1_link = ListNode.array_to_linked_lists_2d(list1)
# print(ListNode.linked_list_to_array(merge_k_lists(l1_link)))


""" 二叉树问题 """


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def array_to_binarytree(nums):
        """ 将数组转换为二叉树 """
        if not nums:
            return None
        root = TreeNode(nums[0])
        queue = [root]
        i = 1
        while i < len(nums):
            current_node = queue.pop(0)
            if nums[i] is not None:
                current_node.left = TreeNode(nums[i])
                queue.append(current_node.left)
            i += 1
            if i < len(nums) and nums[i] is not None:
                current_node.right = TreeNode(nums[i])
                queue.append(current_node.right)
            i += 1
        return root

    @staticmethod
    def preorder_traversal(root):
        """ 使用前序遍历将二叉树转换为数组 """
        if not root:
            return []
        return [root.val] + TreeNode.preorder_traversal(root.left) + TreeNode.preorder_traversal(root.right)


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """
    LeetCode-94.二叉树的中序遍历
    """
    if not root:
        return []
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)


# lists = [1, None, 2, 3]
# print(inorder_traversal(TreeNode.array_to_binarytree(lists)))


def max_depth(root: Optional[TreeNode]) -> int:
    """
    LeetCode-104.二叉树的最大深度
    """
    if not root:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1


# lists = [3, 9, 20, None, None, 15, 7]
# print(max_depth(TreeNode.array_to_binarytree(lists)))


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    LeetCode-226.翻转二叉树
    """
    if not root:
        return
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root


# lists = [4, 2, 7, 1, 3, 6, 9]
# print(TreeNode.preorder_traversal(invert_tree(TreeNode.array_to_binarytree(lists))))


def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    LeetCode-101.对称二叉树
    """
    if not root:
        return True

    def helper(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2 or node1.val != node2.val:
            return False
        return helper(node1.left, node2.right) and helper(node1.right, node2.left)
    return helper(root.left, root.right)


# lists = [1, 2, 2, 3, 4, 4, 3]
# print(is_symmetric(TreeNode.array_to_binarytree(lists)))


def diameter_of_binarytree(root: Optional[TreeNode]) -> int:
    """
    LeetCode-543.二叉树的直径
    """
    def helper(root):
        if not root:
            return 0
        nonlocal res
        left_depth = helper(root.left)
        right_depth = helper(root.right)
        res = max(res, left_depth + right_depth)
        return max(left_depth, right_depth) + 1
    res = 0
    helper(root)
    return res


# lists = [1, 2, 3, 4, 5]
# print(diameter_of_binarytree(TreeNode.array_to_binarytree(lists)))


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    LeetCode-102.二叉树的层序遍历
    """
    if not root:
        return []
    deque = collections.deque()
    deque.append(root)
    res = []
    while deque:
        level = []
        for _ in range(len(deque)):
            node = deque.popleft()
            level.append(node.val)
            if node.left:
                deque.append(node.left)
            if node.right:
                deque.append(node.right)
        res.append(level)
    return res


# lists = [3, 9, 20, None, None, 15, 7]
# print(level_order(TreeNode.array_to_binarytree(lists)))


def sorted_array_to_bst(nums: List[int]) -> Optional[TreeNode]:
    """
    LeetCode-108.将有序数组转换为二叉搜索树
    """
    if not nums:
        return
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    return root


# lists = [-10, -3, 0, 5, 9]
# print(sorted_array_to_bst(lists))


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    LeetCode-98.验证二叉搜索树
    """
    def helper(root, lower=float('-inf'), upper=float('inf')):
        if not root:
            return True
        if not (lower < root.val < upper):
            return False
        return helper(root.left, lower, root.val) and helper(root.right, root.val, upper)
    return helper(root)


# lists = [5, 1, 4, None, None, 3, 6]
# print(is_valid_bst(TreeNode.array_to_binarytree(lists)))


def kth_smallest(root: Optional[TreeNode], k: int) -> int:
    """
    LeetCode-230.二叉搜索树中第k小的元素
    时间复杂度：最坏情况O(n)，平均情况O(k)
    空间复杂度：最坏情况树退化为链表，递归深度为n，空间复杂度O(n)；平均情况为树的高度h，空间复杂度O(h)
    """
    def dfs(root):
        nonlocal k, res
        if not root or k == 0:
            return
        dfs(root.left)
        k -= 1
        if k == 0:
            res = root.val
            return
        dfs(root.right)
    k = k
    res = 0
    dfs(root)
    return res


# lists = [3, 1, 4, None, 2]
# print(kth_smallest(TreeNode.array_to_binarytree(lists), 1))


def right_side_view(root: Optional[TreeNode]) -> List[int]:
    """
    LeetCode-199.二叉树的右视图
    """
    res = []
    d = collections.deque()
    d.append(root)
    while d:
        n = len(d)
        level = []
        for _ in range(n):
            node = d.popleft()
            level.append(node.val)
            if node.left:
                d.append(node.left)
            if node.right:
                d.append(node.right)
        res.append(level[-1])
    return res


# lists = [1, 2, 3, None, 5, None, 4]
# print(right_side_view(TreeNode.array_to_binarytree(lists)))


def flatten(root: Optional[TreeNode]) -> None:
    """
    LeetCode-114.二叉树展开为链表
    """
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    if root.left:
        temp = root.left
        while temp.right:
            temp = temp.right
        temp.right = root.right
        root.right = root.left
        root.left = None


def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    LeetCode-105.从前序与中序遍历序列构造二叉树
    """
    if not inorder or not preorder:
        return None
    root = TreeNode(preorder.pop(0))
    root_index = inorder.index(root.val)
    root.left = build_tree(preorder, inorder[:root_index])
    root.right = build_tree(preorder, inorder[root_index + 1:])
    return root


# list1 = [3, 9, 20, 15, 7]
# list2 = [9, 3, 15, 20, 7]
# print(build_tree(list1, list2))


def path_sum(root: Optional[TreeNode], target_sum: int) -> int:
    """
    LeetCode-437.路径总和Ⅲ
    """
    def dfs(root, current_sum):
        if not root:
            return 0
        res = 0
        current_sum += root.val
        res += prefix_sum[current_sum - target_sum]  # current_sum - prefix_sum = target_sum
        prefix_sum[current_sum] += 1
        res += dfs(root.left, current_sum)
        res += dfs(root.right, current_sum)
        prefix_sum[current_sum] -= 1  # 回溯一步，继续处理其他路径
        return res

    prefix_sum = collections.defaultdict(int)
    prefix_sum[0] = 1
    return dfs(root, 0)


# lists = [10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]
# print(path_sum(TreeNode.array_to_binarytree(lists), 8))


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    LeetCode-236.二叉树的最近公共祖先
    """
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if not left:
        return right
    if not right:
        return left
    return root


# lists = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
# print(lowest_common_ancestor(TreeNode.array_to_binarytree(lists), TreeNode(5), TreeNode(1)))


def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    LeetCode-124.二叉树中的最大路径和
    """
    def helper(root):
        if not root:
            return 0
        nonlocal max_sum
        left_val = max(helper(root.left), 0)
        right_val = max(helper(root.right), 0)
        max_sum = max(max_sum, left_val + right_val + root.val)
        return max(left_val, right_val) + root.val
    max_sum = 0
    helper(root)
    return max_sum


# lists = [-10, 9, 20, None, None, 15, 7]
# print(max_path_sum(TreeNode.array_to_binarytree(lists)))


""" 图论问题 """


def nums_island(grid: List[List[str]]) -> int:
    """
    LeetCode-200.岛屿数量
    """
    def dfs(grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(grid, i - 1, j)
        dfs(grid, i + 1, j)
        dfs(grid, i, j - 1)
        dfs(grid, i, j + 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                count += 1
    return count


# grid = [
#   ["1", "1", "0", "0", "0"],
#   ["1", "1", "0", "0", "0"],
#   ["0", "0", "1", "0", "0"],
#   ["0", "0", "0", "1", "1"]
# ]
# print(nums_island(grid))


def oranges_rotting(grid: List[List[int]]) -> int:
    """
    LeetCode-994.腐烂的橘子
    """
    m, n = len(grid), len(grid[0])
    fresh = 0
    rotten = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                fresh += 1
            elif grid[i][j] == 2:
                rotten.append((i, j))
    if fresh == 0:
        return 0
    minutes = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while rotten and fresh > 0:
        for _ in range(len(rotten)):
            r, c = rotten.pop(0)
            for dr, dc in directions:
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < m and 0 <= c1 < n and grid[r1][c1] == 1:
                    grid[r1][c1] = 2
                    fresh -= 1
                    rotten.append((r1, c1))
        minutes += 1
    return minutes if fresh == 0 else -1


# grid = [[2, 1, 1], [1, 1, 0], [0, 1, 1]]
# print(oranges_rotting(grid))


def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    LeetCode-207.课程表
    """
    graph = collections.defaultdict(list)
    for x, y in prerequisites:
        graph[x].append(y)  # 想要学习课程x，必须先完成课程y
    visited = [0] * num_courses

    def dfs(i):
        if visited[i] == -1:
            return False
        if visited[i] == 1:
            return True
        visited[i] = -1  # 当前课程正在访问中
        for j in graph[i]:
            if not dfs(j):
                return False
        visited[i] = 1  # 当前课程已访问结束
        return True

    for i in range(num_courses):
        if not dfs(i):
            return False
    return True


# lists = [[1, 0], [0, 1]]
# print(can_finish(2, lists))


class Trie:
    """
    LeetCode-208.实现Trie(前缀树)
    """
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def search_prefix(self, prefix: str) -> "Trie":
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def insert(self, word: str) -> None:
        """ 向前缀树中插入字符串word """
        node = self
        for c in word:
            c = ord(c) - ord('a')
            if not node.children[c]:
                node.children[c] = Trie()
            node = node.children[c]
        node.isEnd = True

    def search(self, word: str) -> bool:
        node = self.search_prefix(word)
        return node is not None and node.isEnd

    def starts_with(self, prefix: str) -> bool:
        return self.search_prefix(prefix) is not None


""" 回溯问题 """


def permute(nums: List[int]) -> List[List[int]]:
    """
    LeetCode-46.全排列
    """
    def backtrack(nums, start_index):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if not nums_used[i]:
                nums_used[i] = True
                path.append(nums[i])
                backtrack(nums, start_index + 1)
                path.pop()
                nums_used[i] = False

    res = []
    path = []
    nums_used = [False] * len(nums)
    backtrack(nums, 0)
    return res


# lists = [1, 2, 3]
# print(permute(lists))


def subsets(nums: List[int]) -> List[List[int]]:
    """
    LeetCode-78.子集
    """
    def backtrack(nums, start_index):
        res.append(path[:])
        for i in range(start_index, len(nums)):
            path.append(nums[i])
            backtrack(nums, i + 1)
            path.pop()

    res = []
    path = []
    backtrack(nums, 0)
    return res


# lists = [1, 2, 3]
# print(subsets(lists))


def letter_combinations(digits: str) -> List[str]:
    """
    LeetCode-17.电话号码的字母组合
    """
    if not digits:
        return []
    phone_map = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

    def backtrack(digits, index):
        if index == len(digits):
            res.append(''.join(path))
            return
        number = digits[index]
        letter = phone_map[number]
        for i in range(len(letter)):
            path.append(letter[i])
            backtrack(digits, index + 1)
            path.pop()
    path = []
    res = []
    backtrack(digits, 0)
    return res


# lists = "23"
# print(letter_combinations(lists))


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    LeetCode-39.组合总和
    """
    res = []
    path = []
    candidates.sort()

    def backtrack(candidates, target, start_index):
        if target == 0:
            res.append(path[:])
            return
        for i in range(start_index, len(candidates)):
            if target - candidates[i] < 0:
                break
            path.append(candidates[i])
            backtrack(candidates, target - candidates[i], i)
            path.pop()
    backtrack(candidates, target, 0)
    return res


# lists = [2, 3, 6, 7]
# print(combination_sum(lists, 7))


def generate_parenthesis(n: int) -> List[str]:
    """
    LeetCode-22.括号生成
    """
    if n == 0:
        return ['']
    res = []
    for i in range(n):
        # 将n对括号分解为 i对左括号 + 1个包裹括号 + n-i-1对右括号 的问题
        for left in generate_parenthesis(i):
            for right in generate_parenthesis(n - i - 1):
                res.append('({}){}'.format(left, right))
    return res


# print(generate_parenthesis(2))


def exist(board: List[List[str]], word: str) -> bool:
    """
    LeetCode-79.单词搜索
    """
    def dfs(i, j, k):
        if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
            return False
        if k == len(word) - 1:
            return True
        board[i][j] = ''
        res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
        board[i][j] = word[k]  # 回溯操作
        return res

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(i, j, 0):
                return True
    return False


# board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
# word = "ABCCED"
# print(exist(board, word))


def partition(s: str) -> List[List[str]]:
    """
    LeetCode-131.分割回文串
    """
    def is_palindrome(t):
        if len(t) <= 1:
            return True
        i, j = 0, len(t) - 1
        while i < j:
            if t[i] != t[j]:
                return False
            i += 1
            j -= 1
        return True

    def backtrack(s, start_index):
        if start_index == len(s):
            res.append(path[:])
        for i in range(start_index, len(s)):
            if is_palindrome(s[start_index: i + 1]):
                path.append(s[start_index: i + 1])
                backtrack(s, i + 1)
                path.pop()

    res = []
    path = []
    backtrack(s, 0)
    return res


# strs = 'aab'
# print(partition(strs))


""" 二分查找问题 """


def search_insert(nums: List[int], target: int) -> int:
    """
    LeetCode-35.搜索插入位置
    """
    n = len(nums)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return left


# lists = [1, 3, 5, 6]
# print(search_insert(lists, 7))


def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    LeetCode-74.搜索二维矩阵
    """
    for m in matrix:
        left, right = 0, len(m) - 1
        while left <= right:
            mid = (left + right) // 2
            if m[mid] == target:
                return True
            elif m[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
    return False


def search_matrix_1(matrix: List[List[int]], target: int) -> bool:
    """
    LeetCode-74.搜索二维矩阵
    """
    m, n = len(matrix), len(matrix[0])
    left, right = 0,  m * n - 1
    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid//n][mid%n]
        if mid_val == target:
            return True
        elif mid_val > target:
            right = mid - 1
        else:
            left = mid + 1
    return False


# matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
# print(search_matrix_1(matrix, 19))


def search_range(nums: List[int], target: int) -> List[int]:
    """
    LeetCode-34.在排序数组中查找元素的第一个和最后一个位置
    时间复杂度：当目标值在数组中仅出现一次时，时间复杂度为O(logn)，最坏情况为O(n)
    """
    n = len(nums)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            i, j = mid, mid
            while i >= 0 and nums[i] == target:
                i -= 1
            while j < n and nums[j] == target:
                j += 1
            return [i + 1, j - 1]
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return [-1, -1]


def search_range_1(nums: List[int], target: int) -> List[int]:
    """
    LeetCode-34.在排序数组中查找元素的第一个和最后一个位置
    时间复杂度：时间复杂度为O(logn)
    """
    def find_edge(is_left):
        n = len(nums)
        left, right = 0, n - 1
        edge = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                edge = mid
                if is_left:  # 查找左边界
                    right = mid - 1
                else:  # 查找右边界
                    left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return edge

    left_edge = find_edge(is_left=True)
    right_edge = find_edge(is_left=False)
    return [left_edge, right_edge]


# lists = [5, 7, 7, 8, 8, 10]
# print(search_range_1(lists, 8))


def search(nums: List[int], target: int) -> int:
    """
    LeetCode-33.搜索旋转排序数组
    时间复杂度：二分查找，O(logn)
    """
    n = len(nums)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[left]:  # 说明mid左侧是有序的
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


# lists = [4, 5, 6, 7, 0, 1, 2]
# print(search(lists, 0))


def find_min(nums: List[int]) -> int:
    """
    LeetCode-153.寻找旋转排序数组中的最小值
    """
    n = len(nums)
    left, right = 0, n - 1
    while left < right:  # 若等于时仍进行，因为right=mid会使代码陷入死循环
        mid = (left + right) // 2
        if nums[mid] > nums[right]:  # 说明最小值在mid右侧，且当前值肯定不是最小值
            left = mid + 1
        else:  # 说明最小值在左侧，不确定当前值是不是最小值
            right = mid
    return nums[left]


# lists = [3, 4, 5, 1, 2]
# print(find_min(lists))


def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    LeetCode-4.寻找两个正序数组的中位数
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    n1, n2 = len(nums1), len(nums2)
    left, right = 0, n1
    while left <= right:
        mid1 = (left + right) // 2  # nums1的分割点
        mid2 = (n1 + n2 + 1) // 2 - mid1  # nums2的分割点，确保 左半部分总元素数=（总长度+1）//2
        # 计算分割点边界值，当分割点位于起始或末尾时，用户正负无穷保证比较逻辑的通用性
        max_left1 = float('-inf') if mid1 == 0 else nums1[mid1 - 1]
        min_right1 = float('inf') if mid1 == n1 else nums1[mid1]
        max_left2 = float('-inf') if mid2 == 0 else nums2[mid2 - 1]
        min_right2 = float('inf') if mid2 == n2 else nums2[mid2]
        # 若分割点位置正确，即满足 左半部分 <= 右半部分
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (n1 + n2) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:  # 左半部分太大，左移分割点
            right = mid1 - 1
        else:  # 左半部分太小，右移分割点
            left = mid1 + 1


# lists1 = [1, 2]
# lists2 = [3, 4]
# print(find_median_sorted_arrays(lists1, lists2))

""" 栈相关问题 """


def is_valid(s: str) -> bool:
    """
    LeetCode-20.有效的括号
    时间复杂度：O(n)，对字符串进行一次遍历
    空间复杂度：O(n)，最坏情况下，当输入全是左括号时，栈中会压入n个右括号
    """
    stack = []
    for i in range(len(s)):
        if stack and s[i] == stack[-1]:  # 栈不空且遍历到右括号，且等于栈顶元素
            stack.pop()
        elif s[i] == '(':
            stack.append(')')
        elif s[i] == '[':
            stack.append(']')
        elif s[i] == '{':
            stack.append('}')
        else:  # 遍历到右括号，但栈为空或与栈顶元素不相等，直接返回False
            return False
    return not stack  # 返回栈是否为空，栈为空则代表所有括号都被匹配了，不为空则代表栈中尚有未匹配的左括号


# string = '('
# print(is_valid(string))


class MinStack:
    """
    LeetCode-155.最小栈
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or self.min_stack[-1] >= val:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.min_stack[-1]


def decode_string(s: str) -> str:
    """
    LeetCode-394.字符串解码
    """
    # stack保存之前的状态；res保存当前结果；multi记录当下倍数
    stack: List[List[Union[int, str]]] = []
    res, multi = '', 0
    for c in s:
        if c == '[':
            stack.append([multi, res])
            res, multi = '', 0
        elif c == ']':
            current_multi, last_res = stack.pop()
            res = last_res + current_multi * res
        elif '0' <= c <= '9':
            multi = multi * 10 + int(c)
        else:
            res += c
    return res


# string = "3[a]2[bc]"
# print(decode_string(string))


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    LeetCode-739.每日温度
    单调栈解法
    时间复杂度：O(n)，对数组进行一次遍历
    空间复杂度：O(n)，维护一个单调栈和一个保存结果的数组（大小为n）
    """
    n = len(temperatures)
    res = [0] * n
    stack = [0]  # 存储尚未找到下一次更高温度的天的下标
    for i in range(1, n):
        while stack and temperatures[i] > temperatures[stack[-1]]:  # 栈不空且当前温度大于栈顶元素对应的温度
            idx = stack.pop()
            res[idx] = i - idx  # 计算间隔天数
        stack.append(i)  # 无论哪种情况，最终都需要将当前下标压入栈中
    return res


# lists = [73, 74, 75, 71, 69, 72, 76, 73]
# print(daily_temperatures(lists))


def largest_rectangle_area(heights: List[int]) -> int:
    """
    LeetCode-84.柱状图中最大的矩形
    """
    heights = [0] + heights + [0]  # 添加哨兵节点
    stack = []
    max_area = 0
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            left = stack[-1] if stack else -1  # 左边界为栈顶元素
            width = i - left - 1  # 正确宽度计算
            max_area = max(max_area, h * width)
        stack.append(i)
    return max_area


# lists = [2, 1, 5, 6, 2, 3]
# print(largest_rectangle_area(lists))

""" 堆相关问题 """


def find_kth_largest(nums: List[int], k: int) -> int:
    """
    LeetCode-215.数组中的第K个最大的数字
    时间复杂度：每次递归仅处理包含目标元素的分区，而非像快速排序那样处理两侧。平均时间复杂度为O(n)
    """
    def quick_sort(nums, k):
        pivot_val = random.choice(nums)
        big, equal, small = [], [], []
        for num in nums:
            if num > pivot_val:
                big.append(num)
            elif num < pivot_val:
                small.append(num)
            else:
                equal.append(num)
        if k <= len(big):  # 从大到小取值超不过big数组范围，第k大元素在big数组中
            return quick_sort(big, k)
        if k > len(nums) - len(small):  # 从大到小取值超过了big和equal数组，则第k大元素在small数组中
            return quick_sort(small, k - len(big) - len(equal))
        return pivot_val
    return quick_sort(nums, k)


# lists = [3, 2, 1, 5, 6, 4]
# print(find_kth_largest(lists, 2))


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    LeetCode-347.前k个高频元素
    时间复杂度：统计频率O(n)，构建桶O(n)，收集结果O(n)，总时间复杂度O(n)
    """
    number_freq = collections.Counter(nums)
    max_freq = max(number_freq.values())
    # 桶排序：索引为频率，桶内为对应频率的元素
    bucket = [[] for _ in range(max_freq + 1)]
    for num, freq in number_freq.items():
        bucket[freq].append(num)
    res = []
    for i in range(max_freq, 0, -1):
        if bucket[i]:
            res.extend(bucket[i])
        if len(res) >= k:
            break
    return res[:k]  # 题目保证答案唯一


def top_k_frequent_1(nums: List[int], k: int) -> List[int]:
    """
    LeetCode-347.前k个高频元素
    时间复杂度：
    """
    number_freq = collections.Counter(nums)  # 出现次数
    number = list(number_freq.keys())  # 去除次数后的元素本身

    def quick_sort(candidates, freq, target_k):
        if not candidates:
            return []

        # 三路分区：按频率分为大、等、小
        pivot = random.choice(candidates)
        pivot_freq = freq[pivot]
        big = [num for num in candidates if freq[num] > pivot_freq]
        equal = [num for num in candidates if freq[num] == pivot_freq]
        small = [num for num in candidates if freq[num] < pivot_freq]

        # 递归选择
        if target_k <= len(big):  # 按频率从大到小取，前k个高频元素在big内
            return quick_sort(big, freq, target_k)
        elif target_k <= len(big) + len(equal):  # 按频率从大到小取，前k个高频元素在big + equal内
            return big + equal[:target_k - len(big)]
        else:  # 按频率从大到小取，前k个高频元素在big + equal + small内
            return big + equal + quick_sort(small, freq, target_k - len(big) - len(equal))

    return quick_sort(number, number_freq, k)


# lists = [1, 1, 1, 2, 2, 3]
# print(top_k_frequent_1(lists, 2))


""" 贪心算法问题 """


def can_jump(nums: List[int]) -> bool:
    """
    LeetCode-55.跳跃游戏
    """
    max_jump = 0
    for i in range(len(nums)):
        if max_jump < i:
            return False
        if max_jump >= len(nums) - 1:
            return True
        max_jump = max(max_jump, i + nums[i])


# lists = [2, 3, 1, 1, 4]
# print(can_jump(lists))


def jump(nums: List[int]) -> int:
    """
    LeetCode-45.跳跃游戏Ⅱ
    """
    max_jump = 0
    last_jump = 0
    step = 0
    for i in range(len(nums) - 1):
        if max_jump >= i:
            max_jump = max(max_jump, i + nums[i])
            if i == last_jump:
                step += 1
                last_jump = max_jump
    return step


# lists = [2, 3, 1, 1, 4]
# print(jump(lists))


""" 动态规划问题 """


def climb_stairs(n: int) -> int:
    """
    LeetCode-70.爬楼梯
    动态规划+空间复杂度优化
    时间复杂度：O(n)，遍历一遍数组
    空间复杂度：O(1)，仅使用常数个变量
    """
    a, b = 0, 1  # 初始化
    for i in range(n):
        a, b = b, a + b  # 状态转移
    return b


# print(climb_stairs(3))


def generate(num_rows: int) -> List[List[int]]:
    """
    LeetCode-118.杨辉三角
    时间复杂度：O(n^2)，双重循环
    空间复杂度：O(n^2)，维护一个二维数组存储最终结果
    """
    res = [[1]]  # 存储结果
    for i in range(1, num_rows):
        level = [1] + [0] * (i - 1) + [1]  # 每一行的结果
        for j in range(1, len(level) - 1):  # 从第二个数到倒数第二个数
            level[j] = res[-1][j - 1] + res[-1][j]
        res.append(level)  # 添加每行结果
    return res


# print(generate(3))


def rob(nums):
    """
    LeetCode-198.打家劫舍
    动态规划：每个房间有偷或不偷两种选择，定义dp[i]为前i个房间能偷到的最大金额
    时间复杂度：O(n)，只需遍历一遍房间金额数组
    空间复杂度：O(n)，使用了一个大小为n的数组存储能偷到的最大金额
    """
    n = len(nums)
    if n == 0:
        return 0
    elif n == 1:
        return nums[0]
    dp = [0] * n
    # 初始化：前一个房间能获得的最大金额为nums[0]，前两个房间能偷到的最大金额为max(nums[0], nums[1])
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    # 从第三个房间开始
    for i in range(2, n):
        # 在偷或不偷当前房间情况下，取较大者
        # 偷当前房间，则前一个房间不能偷，即排除掉前一个房间再加上当前房间金额，dp[i - 2] + nums[i]
        # 不偷当前房间，则可以偷前一个房间，能获得的最大金额为dp[i - 1]
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    return dp[-1]


# lists = [2, 7, 9, 3, 1]
# print(rob(lists))


def rob_1_1(nums: List[int]) -> int:
    """
    LeetCode-198.打家劫舍：房间呈线性排列，相邻的房间不能偷
    动态规划：对空间复杂度进行优化，因为当前房间的状态只依赖于前两个房间的状态
    时间复杂度：O(n)，只需遍历一遍房间金额数组
    空间复杂度：O(1)，仅使用常数个变量存储结果
    """
    n = len(nums)
    if n == 0:
        return 0
    elif n == 1:
        return nums[0]
    # 初始化：前一个房间能获得的最大金额为nums[0]，前两个房间能偷到的最大金额为max(nums[0], nums[1])
    dp_0 = nums[0]
    dp_1 = max(nums[0], nums[1])
    # 从第三个房间开始
    for i in range(2, n):
        # 在偷或不偷当前房间情况下，取较大者
        # 偷当前房间，则前一个房间不能偷，即排除掉前一个房间再加上当前房间金额，dp[i - 2] + nums[i]
        # 不偷当前房间，则可以偷前一个房间，能获得的最大金额为dp[i - 1]
        current = max(dp_0 + nums[i], dp_1)
        dp_0, dp_1 = dp_1, current
    return dp_1


def num_squares(n: int) -> int:
    """
    LeetCode-279.完全平方数
    动态规划-完全背包问题：物品是所有小于n的完全平方数，背包是目标值n
    时间复杂度:外层循环O(n)，内层循环O(sqrt(n))，总时间复杂度O(n×sqrt(n))
    空间复杂度：O(n)，维护一个大小为n的dp数组
    """
    # dp[i]定义为和为i时所需的最少完全平方数数量
    dp = [0] + [float('inf')] * n  # 初始化dp[0]=0
    for i in range(1, n + 1):  # 外层遍历物品
        j = 1
        while j * j <= i:  # 对于当前值i，尝试使用所有可能的平方数j*j去组合
            # 取不使用当前数字j 和 使用情况下：i-j*j的最优解加一（即多一个j） 的最小值
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[-1]


# print(num_squares(12))


def coin_change(coins: List[int], amount: int) -> int:
    """
    LeetCode-322.零钱兑换
    完全背包问题->物品(硬币)无限，用物品(硬币)去装满(凑)背包(总金额)
    时间复杂度：外层循环遍历n种硬币，时间复杂度为O(n);内层循环遍历amount次，时间复杂度为O(amount)
              总时间复杂度为O(n×amount)
    空间复杂度：使用一个大小为amount + 1的dp数组存储中间状态，故空间复杂度为O(amount)
    """
    # dp[i]定义为凑出金额i所需的最少硬币数量
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 初始化：凑出金额0，最少需0个硬币
    for coin in coins:  # 遍历物品
        for i in range(coin, amount + 1):  # 正向遍历背包
            dp[i] = min(dp[i], dp[i - coin] + 1)  # 不使用当前硬币 或 使用当前硬币（即凑成金额i-coin的最少硬币数加一）
    return dp[-1] if dp[-1] != float('inf') else -1


# lists = [1, 2, 5]
# print(coin_change(lists, 11))


def word_break(s: str, word_dict: List[str]) -> bool:
    """
    LeetCode-139.单词拆分
    动态规划-完全背包问题：字符串中每个字符是物品，目标单词是背包
    时间复杂度：外层循环O(n)，内层循环最坏也是O(n)，故总时间复杂度O(n^2)
    空间复杂度：O(n)，维护一个大小为n的dp数组
    """
    # dp[i]定义为字符串s的前i个字符是否可以被拆分为字典中的单词组合
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # 初始化，表示空字符串默认可以拆分
    for i in range(1, n + 1):  # 遍历物品
        for j in range(i):  # 遍历背包
            # 若存在某个分割点j，在dp[j]=True的情况下，s[j:i]也是字典中的单词，则dp[i]可以被拆分
            if dp[j] and s[j: i] in word_dict:
                dp[i] = True
                break
    return dp[n]


# string = 'leetcode'
# wordDict = ["leet", "code"]
# print(word_break(string, wordDict))

def length_of_lis(nums):
    """
    LeetCode-300.最长递增子序列：给一个数组，找其中最长严格递增子序列的长度
    动态规划解法：定义dp[i]含义为以下标i结尾的子数组的最长严格递增子序列的长度
    时间复杂度：外层循环遍历数组的每个元素，时间复杂度为O(n)；内层循环遍历每个元素之前的所有元素，最坏情况下时间复杂度为O(n)
              故总的时间复杂度为O(n^2)
    空间复杂度：O(n)，使用了长度为n的dp数组来存储每个位置的结果
    """
    n = len(nums)
    dp = [1] * n  # 初始化，每个子数组的最长严格递增子序列的长度至少为1
    for i in range(1, n):
        for j in range(i):  # 遍历元素nums[i]之前的所有元素
            if nums[i] > nums[j]:
                # 若nums[i]大于nums[j]，则说明以nums[j]结尾的数组的最长递增子序列dp[j]加上nums[i]可以构成一个更长的递增序列
                dp[i] = max(dp[j] + 1, dp[i])  # 与dp[i]比较的原因在于可能存在多个nums[j]去更新dp[i]
    return dp[-1]


# lists = [10, 9, 2, 5, 3, 7, 101, 18]
# print(length_of_lis(lists))


def length_of_lis_1(nums):
    """
    LeetCode-300.最长递增子序列：给一个数组，找其中最长严格递增子序列的长度
    单调栈解法：维护一个"潜在最长子序列"的栈，尽可能减小后续元素的递增阈值，从而为更长的子序列留出空间
    时间复杂度：外层循环遍历所有元素，时间复杂度为O(n)；
              内层二分查找，查找范围为栈的长度，最坏情况下栈的长度等于数组长度，时间复杂度为O(logn);
              因此，总时间复杂度为O(nlogn)
    空间复杂度：O(n)，最坏情况下(数组完全递增)，栈会存储所有元素，即O(n)
    """
    stack = [nums[0]]
    for i in range(1, len(nums)):
        # 若当前值大于等于单调栈的最后一个元素值，则直接加入单调栈中
        if nums[i] > stack[-1]:
            stack.append(nums[i])
        # 在已有的单调递增栈中找到第一个大于等于当前值的元素，然后替换掉该元素
        else:
            # 使用二分查找法，在查找过程中更新loc，直到查找结束即可
            left, right = 0, len(stack) - 1
            while left < right:
                mid = (left + right) // 2
                if stack[mid] >= nums[i]:
                    right = mid  # mid是一个满足条件的值，但可能会有更小的值
                else:
                    left = mid + 1
            stack[right] = nums[i]  # 交换位置
    return len(stack)


def max_product(nums: List[int]) -> int:
    """
    LeetCode-152.乘积最大子数组
    动态规划：维护两个变量max_val和min_val，分别记录以当前元素结尾的最大乘积和最小乘积
    时间复杂度：O(n)，仅遍历一遍数组
    空间复杂度：O(1)，仅维护常数个变量
    """
    n = len(nums)
    res = max_val = min_val = nums[0]  # 初始化
    for i in range(1, n):
        if nums[i] < 0:  # 当前值为负时交换最大最小值
            max_val, min_val = min_val, max_val
        max_val = max(nums[i], nums[i] * max_val)  # 更新最大值
        min_val = min(nums[i], nums[i] * min_val)  # 更新最小值
        res = max(res, max_val)  # 更新全局最大值
    return res


# lists = [2, 3, -2, 4]
# print(max_product(lists))


def can_partition(nums: List[int]) -> bool:
    """
    LeetCode-416.分割等和子集
    动态规划-0-1背包问题：目标和为背包，数组中元素为物品
    时间复杂度：O(n×target)，外层循环遍历数组O(n)，内层循环O(target)
    空间复杂度：O(target)，维护一个大小为target的一维dp数组
    """
    total = sum(nums)
    if total % 2 != 0:  # 若数组和为奇数则必然无法分割成两个等和子集
        return False
    target = total // 2
    dp = [False] * (target + 1)  # dp[i]表示是否存在一个和为target的子集
    dp[0] = True  # 初始化，和为0的子集总是存在（空集）
    for num in nums:  # 遍历物品
        # 逆序遍历背包，当背包容量小于当前元素num时截至
        # 表示背包容量都小于当前物品体积了必然不可能
        for i in range(target, num - 1, -1):
            # 前一步已经存在和为i的子集 或 存在和为i-num的子集(加上当前num后和为i)
            dp[i] = dp[i] or dp[i - num]
    return dp[-1]


# lists = [1, 5, 11, 5]
# print(can_partition(lists))

def longest_valid_parentheses(s: str) -> int:
    """
    LeetCode-32.最长有效括号
    栈的解法
    时间复杂度：O(n)，只遍历一遍
    空间复杂度：O(n)，最坏情况下输入全为左括号
    """
    stack = [-1]  # 栈的最左边始终记录 最后一个无法匹配的右括号索引，初始为-1用于处理边界条件
    max_length = 0
    for i in range(len(s)):
        if s[i] == '(':  # 遇到左括号，直接添加索引，可能成为后续右括号的匹配起点
            stack.append(i)
        else:  # 遇到右括号
            stack.pop()  # 若栈顶为左括号则成功匹配一对括号，若栈顶是基准索引则表示当前右括号无法被匹配
            if not stack:  # 若栈为空，则当前右括号无法匹配，成为新基准
                stack.append(i)
            else:  # 成功匹配，更新最大长度
                max_length = max(max_length, i - stack[-1])
    return max_length


def longest_valid_parentheses_1(s: str) -> int:
    """
    LeetCode-32.最长有效括号
    动态规划解法
    时间复杂度：O(n)，遍历一遍字符串
    空间复杂度：O(n)，维护一个长为n的dp数组
    """
    dp = [0] * len(s)
    max_length = 0
    for i in range(1, len(s)):
        if s[i] == ')':  # 遇到右括号
            if s[i - 1] == '(':  # 且前一个字符是左括号
                dp[i] = dp[i - 2] + 2 if i >= 2 else 2  # 有效长度+2
            else:  # 前一个字符是右括号
                j = i - dp[i - 1] - 1  # 跳过已匹配的子串，找到与当前右括号匹配的左括号位置
                if j >= 0 and s[j] == '(':  # 若找到的位置有效且就是左括号，则长度＋2
                    dp[i] = dp[i - 1] + 2
                    if j >= 1:  # 处理连续有效子串的合并，若j>=1且所处位置是左括号，则说明可能在这之前仍存在有效括号
                        dp[i] += dp[j - 1]  # 前后有效括号长度合并
            max_length = max(max_length, dp[i])  # 更新最大长度
    return max_length


# string = ')()())'
# print(longest_valid_parentheses_1(string))


""" 多为动态规划问题 """


def unique_paths(m: int, n: int) -> int:
    """
    LeetCode-62.不同路径
    时间复杂度：O(m×n)
    空间复杂度：O(m×n)
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


def unique_paths_1(m: int, n: int) -> int:
    """
    LeetCode-62.不同路径
    时间复杂度：O(m×n)，双重循环
    空间复杂度：O(n)，仅维护大小为n的dp数组
    """
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j - 1]
    return dp[-1]

# print(unique_paths(3, 7))


def min_path_sum(grid: List[List[int]]) -> int:
    """
    LeetCode-64.最小路径和
    时间复杂度：O(m×n)，双重循环
    空间复杂度：O(m×n)，维护一个二维dp数组
    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):  # 初始化第一列
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for i in range(1, n):  # 初始化第一行
        dp[0][i] = dp[0][i - 1] + grid[0][i]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]


def min_path_sum_1(grid: List[List[int]]) -> int:
    """
    LeetCode-64.最小路径和
    时间复杂度：O(m×n)，双重循环
    空间复杂度：O(1)，原地修改
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if i == j == 0:
                continue
            elif i == 0:  # 初始化第一行
                grid[i][j] = grid[i][j - 1] + grid[i][j]
            elif j == 0:  # 初始化第一列
                grid[i][j] = grid[i - 1][j] + grid[i][j]
            else:
                grid[i][j] = min(grid[i][j - 1], grid[i - 1][j]) + grid[i][j]
    return grid[-1][-1]


# grids = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
# print(min_path_sum(grids))


def longest_palindrome(s: str) -> str:
    """
    LeetCode-5.最长回文子串
    中心扩散法
    时间复杂度：O(n^2)，外层循环O(n)，内层回文中心最多向外扩展n次也是O(n)
    空间复杂度：O(1)，仅使用常数个变量
    """
    def palindrome(s, left, right):
        """ 两端扩散，寻找回文子串 """
        while 0 <= left < len(s) and 0 <= right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    n = len(s)
    start, end = 0, 0  # 记录最长回文子串的起始、终止位置
    for i in range(n):
        start1, end1 = palindrome(s, i, i)  # 回文中心为1个字符时的情况
        if end1 - start1 > end - start:
            start, end = start1, end1
        start2, end2 = palindrome(s, i, i + 1)  # 回文中心为2个字符时的情况
        if end2 - start2 > end - start:
            start, end = start2, end2
    return s[start: end + 1]


def longest_palindrome_1(s: str) -> str:
    """
    LeetCode-5.最长回文子串
    动态规划
    时间复杂度：O(n^2)，外层循环O(n)，内层回文中心最多向外扩展n次也是O(n)
    空间复杂度：O(n^2)，维护一个二维dp数组
    """
    n = len(s)
    if n < 2:
        return s
    max_len = 1  # 记录最长回文子串的长度
    start = 0  # 记录最长回文子串的起始索引
    # dp[i][j]表示s[i:j]是否是回文串
    dp = [[False] * n for _ in range(n)]
    for i in range(n):  # 初始化所有长度为1的子串都是回文串
        dp[i][i] = True
    for l in range(2, n + 1):  # 子串长度从2到n
        for i in range(n):  # 遍历左边界
            j = l + i - 1  # 根据子串长度和左边界计算右边界
            if j >= n:  # 若右边界越界，则退出当前循环
                break
            if s[i] != s[j]:  # 若左右边界对应的字符不同，则当前子串不是回文串
                dp[i][j] = False
            else:  # 若左右边界对应的字符相同
                if j - i < 3:  # 若当前回文串长度为2或3
                    dp[i][j] = True
                else:  # 若当前回文串长度大于3
                    dp[i][j] = dp[i + 1][j - 1]
            if dp[i][j] and j - i + 1 > max_len:  # 更新最长回文串的长度和起始索引
                max_len = j - i + 1
                start = i
    return s[start: start + max_len]


# string = "babad"
# print(longest_palindrome_1(string))


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    LeetCode-1143.最长公共子序列
    动态规划
    时间复杂度：O(n1×n2),双重循环遍历
    空间复杂度：O(n1×n2),维护一个二维dp数组
    """
    n1, n2 = len(text1), len(text2)
    # text1中以text1[i-1]结尾的字符串 和 text2中以text2[j-1]结尾的字符串 的最长公共子序列
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if text1[i - 1] == text2[j - 1]:  # 若字符相同，最长公共序列长度+1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:  # 若字符不同，则最长公共序列长度为
                # text1[0:i - 1]和text2[0:j]的最长公共序列长度 和 text1[0:i]和text2[0:j - 1]的最长公共序列长度 中的较大值
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# print(longest_common_subsequence('abcde', 'ace'))


def min_distance(word1: str, word2: str) -> int:
    """
    LeetCode-72.编辑距离
    动态规划
    时间复杂度：O(n1×n2),双重循环遍历
    空间复杂度：O(n1×n2),维护一个二维dp数组
    """
    n1, n2 = len(word1), len(word2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):  # 初始化第一列，转换成空字符所需最少操作数
        dp[i][0] = i
    for i in range(n2 + 1):  # 初始化第一行，转换成空字符所需最少操作数
        dp[0][i] = i
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 两个单词分别添加/删除其中一个字符；替换其中一个单词的第i/j个字符
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


# print(min_distance('horse', 'ros'))


""" 技巧性问题 """


def single_number(nums: List[int]) -> int:
    """
    LeetCode-136.只出现一次的数字
    位运算：异或运算->两个相同数字异或为0
        若将数组中所有数字执行异或运算，则最后留下的结果一定为只出现一次的数字
    """
    x = 0
    for num in nums:
        x ^= num
    return x


# lists = [2, 2, 1]
# print(single_number(lists))


def majority_element(nums: List[int]) -> int:
    """
    LeetCode-169.多数元素
    摩尔投票：记输入数组的众数为x,长度为n。
        若记众数的票数为+1，非众数的票数为-1，则一定有所有数字的票数和>0;
        若数组前a个数字的票数和=0，则剩余n-a个数字的票数和一定仍 >0，即后n-a个数字的众数仍为x
    """
    votes = 0
    x = 0
    for num in nums:
        if votes == 0:
            x = num
        votes += 1 if num == x else -1
    return x


# lists = [2, 2, 1, 1, 1, 2, 2]
# print(majority_element(lists))


def sort_colors(nums: List[int]) -> List[int]:
    """
    LeetCode-75.颜色分类
    """
    n = len(nums)
    p = 0
    for i in range(n):
        if nums[i] == 0:
            nums[p], nums[i] = nums[i], nums[p]
            p += 1

    for i in range(n):
        if nums[i] == 1:
            nums[p], nums[i] = nums[i], nums[p]
            p += 1
    return nums


# lists = [2, 0, 2, 1, 1, 0]
# print(sort_colors(lists))


def next_permutation(nums: List[int]) -> List[int]:
    """
    LeetCode-31.下一个排列
    """
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:  # 从后向前查找第一个[较小的数]，这个数要尽量靠右
        i -= 1
    if i >= 0:
        j = n - 1
        while j >= 0 and nums[i] >= nums[j]:  # 从后向前查找第一个[较大的数]，这个数要尽量小
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    left, right = i + 1, n - 1
    while left < right:  # 使用双指针反转较小数右侧的子数组使其变为升序排列
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    return nums


# lists = [1, 2, 3]
# print(next_permutation(lists))


def find_duplicate(nums: List[int]) -> int:
    """
    LeetCode-287.寻找重复数
    """
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow


def find_duplicate_1(nums: List[int]) -> int:
    """
    LeetCode-287.寻找重复数
    """
    # 在数字范围[1,n]内进行二分查找
    left, right = 1, len(nums) - 1  # 数字范围为 [1, n]
    while left < right:
        mid = (left + right) // 2
        count = 0
        # 统计数组中小于mid的元素个数
        for num in nums:
            if num <= mid:
                count += 1
        if count > mid:  # 若元素数量大于mid,重复数在左半部分
            right = mid
        else:  # 重复数在右半部分
            left = mid + 1
    return left


# lists = [1, 3, 4, 2, 2]
# print(find_duplicate_1(lists))

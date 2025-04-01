from typing import List, Optional, Union
from Top_100 import TreeNode
""" 打家劫舍问题 """


def rob_1(nums: List[int]) -> int:
    """
    LeetCode-198.打家劫舍：房间呈线性排列，相邻的房间不能偷
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


# money = [2, 7, 9, 3, 1]
# print(rob_1_1(money))


def rob_2(nums: List[int]) -> int:
    """
    LeetCode-213.打家劫舍Ⅱ：房间呈环形排列，不能同时偷相邻房间和首尾房间
    动态规划：分割房间，考虑两种情况，偷前n-1个房间和后n-1个房间，分别求所能偷到的最大金额，然后取较大值
    时间复杂度：O(n)，只需遍历一遍房间金额数组
    空间复杂度：O(n)/O(1)，使用了一个大小为n的数组存储能偷到的最大金额 或 使用常数个变量
    """
    n = len(nums)
    if n == 0:
        return 0
    elif n == 1:
        return nums[0]
    else:
        return max(rob_1(nums[1:]), rob_1(nums[:-1]))  # 直接调用-打家劫舍Ⅰ-的解法


# money = [2, 7, 9, 3, 1]
# print(rob_2(money))


def rob_3(root: Optional[TreeNode]) -> int:
    """
    LeetCode-337.打家劫舍Ⅲ：房屋（节点）呈二叉树形排列，不能偷相邻房间（节点）
    递归+动态规划：对于每个节点，都有 偷或不偷 两种选择，若偷当前节点，则左右子节点不能偷，反之可以偷
        递归返回两种情况，即偷和不偷当前节点能获得的最大金额，[rob, not_rob]
    时间复杂度：O(n)，每个节点访问了一次
    空间复杂度：O(n)，递归调用的栈空间，树退化为链表时最坏情况下是O(n)，且每个递归返回一个长度为2的数组
    """
    def rob_helper(node: Optional[TreeNode]):
        if not node:
            return [0, 0]
        left = rob_helper(node.left)
        right = rob_helper(node.right)
        # 偷当前节点：当前节点值+不偷左右子节点的最大金额
        rob = node.val + left[1] + right[1]
        # 不偷当前节点：左右子节点能获得的最大金额之和
        not_rob = max(left) + max(right)
        return [rob, not_rob]  # 每次递归的返回值
    return max(rob_helper(root))


def rob_4(nums: List[int], k: int) -> int:
    """
    LeetCode-2560.打家劫舍Ⅳ
    最小化最大值问题——二分查找 + 贪心验证
    时间复杂度：O(nlogm),其中n是数组长度，m是数组最大值与最小值的差值
             遍历数组的时间复杂度是O(n)，二分查找的时间复杂度是O(logm)
    空间复杂度：O(1)，仅使用常数个变量
    """
    def check(y):
        """
        在最大金额y的限制下，小偷最多可以偷取的房屋数量
        贪心策略——>能偷则偷
        """
        count, visited = 0, False
        for x in nums:
            # 当房屋金额小于等于y且前一个房间没有被偷时，偷取当前房屋
            if x <= y and not visited:
                count += 1
                visited = True  # 标记已偷
            else:
                visited = False  # 重置标记
        return count

    #  能偷到的最小金额肯定处于[min(nums), max(nums)]之内
    left, right = min(nums), max(nums)
    while left < right:
        mid = (left + right) // 2
        if check(mid) >= k:  # mid是一个可行解，但是也可能存在更小的mid
            right = mid  # 保留mid在可行解范围内
        else:
            left = mid + 1  # 将mid排除出可行解范围
    return left  # 当left=right时，check(left)必定>=k，即left指向的必然是最小的可行解


lists = [3, 1, 2, 4]
print(rob_4(lists, 2))

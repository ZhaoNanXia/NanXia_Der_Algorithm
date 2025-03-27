from typing import List, Optional, Union


""" 0-1背包问题 """


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


def last_stone_weight(stones: List[int]) -> int:
    """
    LeetCode-1049.最后一块石头的重量Ⅱ
    动态规划-0-1背包问题：目标值为背包，元素为物品
    时间复杂度：O(n×target)，外层循环遍历数组O(n)，内层循环O(target)
    空间复杂度：O(target)，维护一个大小为target+1的一维dp数组
    """
    total = sum(stones)
    target = total // 2  # 将原问题转换为寻找最接近total/2的子集和
    dp = [False] * (target + 1)  # dp[i]定义为能否凑成和为i的子集
    dp[0] = True  # 初始化
    for w in stones:  # 遍历物品
        for i in range(target, w - 1, -1):  # 遍历背包
            dp[i] = dp[i] or dp[i - w]  # 选dp[i-w]或不选dp[i]，只要一个为真即可

    for i in range(target, -1, -1):  # 倒序遍历寻找最接近total/2的数值
        if dp[i]:
            return total - 2 * i  # 返回最小重量


def find_target_sum_ways(nums: List[int], target: int) -> int:
    """
    LeetCode-494.目标和
    动态规划-0-1背包问题：记数组的元素和为total,添加‘-’的元素之和为neg,则其余添加‘+’的元素和为pos;
        则有pos+neg=total,pos-neg=target,结合上述两式可得：pos=(total+target)//2
        问题转化为给定一个大小为pos的背包，从数组中选择若干元素（物品）装满背包。
    时间复杂度：O(n×target_sum)，外层循环遍历数组O(n)，内层循环O(target_sum)
    空间复杂度：O(target_sum)，维护一个大小为target_sum+1的一维dp数组
    """
    total = sum(nums)
    if abs(target) > total:
        return 0
    if (target + total) % 2 == 1:
        return 0
    target_sum = (target + total) // 2
    dp = [0] * (target_sum + 1)  # dp[i]定义为和为i的表达式数目
    dp[0] = 1  # 初始化和为0的情况只有一种
    for num in nums:  # 遍历物品
        for j in range(target_sum, num - 1, -1):  # 遍历背包
            dp[j] += dp[j - num]  # 和为j-num的数量加上当前元素num刚好和为j
    return dp[target_sum]


def find_max_form(strs: List[str], m: int, n: int) -> int:
    """
    LeetCode-474.一和零
    动态规划-0-1背包问题：字符串中元素是物品，但是需满足两种背包容量限制：m和n。
    时间复杂度：O(n×m×n)，外层循环遍历字符串O(n)，内层循环O(m*n)
    空间复杂度：O(m×n)，维护一个大小为m×n的二维dp数组
    """
    # dp[i][j]表示最多使用i个0和j个1的情况下，最大子集的大小
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for c in strs:
        ones = c.count('1')
        zeros = c.count('0')
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                # 不选当前字符串 或 选当前字符串（选之前的最优解加上当前字符串）
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[-1][-1]


""" 完全背包问题 """


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
            # 不使用当前硬币 或 使用当前硬币（即凑成金额i-coin的最少硬币数加一）
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


# lists = [1, 2, 5]
# print(coin_change(lists, 11))

def combination_sum4(nums: List[int], target: int) -> int:
    """
    LeetCode-377.组合总和Ⅳ
    完全背包问题->目标和为背包，元素为物品。
        因为原题要求统计的是排列数，故先遍历背包确保一定背包容量下所有物品都能被独立选择，形成不同的排列顺序。
    时间复杂度：O(target×n)，两层循环，外层O(target)，内层O(n)
    空间复杂度：O(target)，维护一个大小为target+1的dp数组
    """
    dp = [0] * (target + 1)  # dp[i]定义为和为i的组合数量
    dp[0] = 1
    for i in range(target + 1):  # 遍历背包
        for num in nums:  # 遍历物品
            if num <= i:
                # 和为i-num的组合数量加上当前元素num刚好和为i
                dp[i] += dp[i - num]
    return dp[-1]


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

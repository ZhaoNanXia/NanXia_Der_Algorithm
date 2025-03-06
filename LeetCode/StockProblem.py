""" 股票问题 """


def max_profit_1(prices):
    """
    LeetCode-121.买卖股票的最佳时机：只能交易一次
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),只使用了常数个变量存储中间状态
    """
    max_profit = float('-inf')  # 记录最低价格
    min_price = float('inf')  # 记录最大利润
    for price in prices:
        min_price = min(min_price, price)  # 更新最低价格
        max_profit = max(max_profit, price - min_price)  # 更新最大利润
    return max_profit


def max_profit_2(prices):
    """
    LeetCode-122. 买卖股票的最佳时机II：可以交易多次，但每次最多只能持有一股
    动态规划解法：定义每天的状态为 持有 或 不持有 两种
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(n),使用一个大小为n×2的二维数组存储每天的状态
    """
    n = len(prices)
    # 定义每天的状态为 持有股票(1) 和 不持有股票(0) 两种情况
    dp = [[0] * 2 for _ in range(n)]
    # 初始化第一天的状态：不持有利润为0，持有利润为-prices[0]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, n):
        # 不持有：前一天不持有 或 前一天持有当天卖出
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        # 持有：前一天持有 或 当天买入
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[-1][0]


def max_profit_2_1(prices):
    """
    LeetCode-122. 买卖股票的最佳时机II：可以交易多次，但每次最多只能持有一股
    动态规划解法：优化空间复杂度，因此每天的状态只依赖于前一天的状态
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),只使用了常数个变量存储中间状态
    """
    n = len(prices)
    # 初始化第一天的状态：不持有利润为0，持有利润为-prices[0]
    dp_0 = 0
    dp_1 = -prices[0]
    for i in range(1, n):
        # 不持有：前一天不持有 或 前一天持有当天卖出
        dp_0 = max(dp_0, dp_1 + prices[i])
        # 持有：前一天持有 或 当天买入
        dp_1 = max(dp_1, dp_0 - prices[i])
    return dp_0


# nums = [7, 1, 5, 3, 6, 4]
# print(max_profit_2_1(nums))


def max_profit_2_2(prices):
    """
    LeetCode-122. 买卖股票的最佳时机II：可以交易多次，但每次最多只能持有一股
    贪心解法: 从第二天开始，只要当天价格比前一天高，就在前一天买入当天卖出
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),只使用了常数个变量存储中间状态
    """
    max_profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            max_profit += prices[i] - prices[i - 1]
    return max_profit


# nums = [7, 1, 5, 3, 6, 4]
# print(max_profit_2_1(nums))


def max_profit_3(prices):
    """
    LeetCode-123. 买卖股票的最佳时机III：最多交易两次
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),只使用了常数个变量存储中间状态
    """
    n = len(prices)
    buy_1 = buy_2 = -prices[0]  # 初始化买入时的最大利润为-prices[0]
    sell_1 = sell_2 = 0  # 初始化卖出时的最大利润为0
    for i in range(1, n):
        buy_1 = max(buy_1, -prices[i])  # 第一次买入：取 之前的最大值和当天买入的值 中的较大者
        sell_1 = max(sell_1, buy_1 + prices[i])  # 第一次卖出：取 之前的最大值和用第一次买入的利润再卖出的值 中的较大者
        buy_2 = max(buy_2, sell_1 - prices[i])  # 第二次买入：取 之前的最大值和用第一次卖出的利润再买入的值 中的较大者
        sell_2 = max(sell_2, buy_2 + prices[i])  # 第二次卖出：取 之前的最大值和用第二次买入的利润再当天卖出的值 中的较大者
    return sell_2


# nums = [7, 1, 5, 3, 6, 4]
# print(max_profit_3(nums))


def max_profit_4(prices, k):
    """
    LeetCode-188. 买卖股票的最佳时机IV：最多可以交易K次，最多同时交易一笔
    动态规划解法
    时间复杂度：O(nk)，两层循环，外循环O(n)，内循环O(k)
    空间复杂度：O(k)，使用了一个大小为2K+1的数组存储每天的状态
    """
    n = len(prices)
    # k次交易便有 k次买入和k次卖出,则每一天都有交易（2k种可能）和不交易（1种可能）两种情况，即2k+1
    # 定义每天的状态有2K+1种情况，索引0代表不交易，奇数索引代表买入，偶数索引代表卖出
    dp = [0] * (2 * k + 1)
    for i in range(1, 2 * k + 1, 2):  # 初始化每一次买入操作
        dp[i] = -prices[0]
    for i in range(1, n):
        for j in range(0, 2 * k - 1, 2):
            # 买入(奇数索引)：对比前一次买入后的收益和此次买入后的收益
            dp[j + 1] = max(dp[j + 1], dp[j] - prices[i])
            # 卖出(偶数索引)：对比前一次卖出后的收益和前一次买入后此次卖出后的收益
            dp[j + 2] = max(dp[j + 2], dp[j + 1] + prices[i])
    return dp[-1]  # 返回最后一天进行了所有交易后的最大利润


# nums = [7, 1, 5, 3, 6, 4]
# print(max_profit_4(nums, 3))


def max_profit_5(prices):
    """
    LeetCode-309. 买卖股票的最佳时机含冷冻期：可以多次交易，但卖出后第二天无法买入
    动态规划解法：定义每天三种状态：
        0：持有股票
        1：在冷冻期而未持有股票（可解释为：在前一天持有的基础上今天卖出了，导致从卖出开始就成为冷冻期，
                                     无法在第二天参与买入操作）
        2：不在冷冻期而未持有股票（可以参与第二天的买入操作）
    时间复杂度：O(n)，遍历一遍价格数组
    空间复杂度：O(n)，使用了一个大小为n×3的二维数组
    """
    n = len(prices)
    dp = [[0] * 3 for _ in range(n)]
    dp[0][0] = -prices[0]  # 初始化第一天持有股票的利润为-prices[0]
    for i in range(1, n):
        # 持有股票：取 前一天的持有值(dp[i - 1][0]) 或
        # 前一天不在冷冻期而未持有，所以当天可以买入的持有值(dp[i - 1][2] - prices[i]) 中的较大值
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
        # 在冷冻期而未持有股票: 只有前一天持有 且 当天卖出 才能使当天卖出开始成为冷冻期，无法参与第二天买入
        # 前一天的其余两种状态皆无法使当天成为冷冻期
        dp[i][1] = dp[i - 1][0] + prices[i]
        # 不在冷冻期而未持有股票：取 前一天不在冷冻期且未持有 和 前一天处于冷冻期而未持有 中的较大值
        dp[i][2] = max(dp[i - 1][1], dp[i - 1][2])
    return max(dp[-1][1], dp[-1][2])  # 返回最后一天未持有股票情况下的最大利润


nums = [1, 2, 3, 0, 2]
print(max_profit_5(nums))


def max_profit_5_1(prices):
    """
    LeetCode-309. 买卖股票的最佳时机含冷冻期：可以多次交易，但卖出后第二天无法买入
    动态规划解法：优化空间复杂度,当天的状态都依赖于前一天的状态
    时间复杂度：O(n)，遍历一遍价格数组
    空间复杂度：O(1)，仅使用常数个变量存储中间状态
    """
    n = len(prices)
    dp_0 = - prices[0]
    dp_1 = 0
    dp_2 = 0
    for i in range(1, n):
        # 先保存前一天的状态
        prev_dp_0 = dp_0
        prev_dp_1 = dp_1
        prev_dp_2 = dp_2

        dp_0 = max(prev_dp_0, prev_dp_2 - prices[i])
        dp_1 = prev_dp_0 + prices[i]
        dp_2 = max(prev_dp_1, prev_dp_2)
    return max(dp_1, dp_2)


# nums = [1, 2, 3, 0, 2]
# print(max_profit_5_1(nums))


def max_profit_6(prices, fee):
    """
    LeetCode-714. 买卖股票的最佳时机含手续费：不限交易次数，但是每交易一次都要付手续费
    可在问题”买卖股票的最佳时机II“基础上优化，只需在每次卖出后扣除手续费即可
    动态规划解法：定义每天的状态为 持有 或 不持有 两种
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(n),使用一个大小为n×2的二维数组存储每天的状态
    """
    n = len(prices)
    dp = [[0] * 2 for _ in range(n)]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, n):
        # 不持有股票：前一天就不持有 或者 前一天持有但是今天卖出了（需要付手续费）
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
        # 持有股票：前一天就持有 或者 前一天不持有但是今天买入了
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[-1][0]


# nums = [1, 3, 2, 8, 4, 9]
# print(max_profit_6(nums, 2))


def max_profit_6_1(prices, fee):
    """
    LeetCode-714. 买卖股票的最佳时机含手续费：不限交易次数，但是每交易一次都要付手续费
    动态规划解法：优化空间复杂度，当天的状态仅依赖于前一天的状态
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),仅使用常数个变量存储中间状态
    """
    n = len(prices)
    # 初始化第一天的状态：不持有利润为0，持有利润为-prices[0]
    dp_0 = 0
    dp_1 = -prices[0]
    for i in range(1, n):
        # 不持有：前一天不持有 或 前一天持有当天卖出
        dp_0 = max(dp_0, dp_1 + prices[i] - fee)
        # 持有：前一天持有 或 当天买入
        dp_1 = max(dp_1, dp_0 - prices[i])
    return dp_0


# nums = [1, 3, 2, 8, 4, 9]
# print(max_profit_6_1(nums, 2))


def max_profit_6_2(prices, fee):
    """
    LeetCode-714. 买卖股票的最佳时机含手续费：不限交易次数，但是每交易一次都要付手续费
    贪心解法：将手续费放在买入的时候计算，记录遍历过程中股票的最低买入价格，然后在当前股票价格高于最低买入价格的时候买入
    时间复杂度：O(n),只需遍历价格数组一遍
    空间复杂度：O(1),仅使用常数个变量存储中间状态
    """
    max_profit = 0
    buy = prices[0] + fee  # 当前股票的最低买入价格
    for i in range(1, len(prices)):
        # 如果当前股票价格+手续费 小于 最低买入价格，那不如以更低的买入价格买入，则更新最低买入价格
        if prices[i] + fee < buy:
            buy = prices[i] + fee
        elif prices[i] > buy:  # 如果当前股票价格高于最低买入价格，则直接以最低买入价格卖出
            max_profit += prices[i] - buy
            # 反悔操作，万一下一天股票价格更高了，那下一天卖出的利润就是prices[i+1]-prices[i]
            # 再加上上一天卖出的利润, 即prices[i+1]-prices[i]+prices[i]-buy=prices[i+1]-buy
            # 恰好相当于在当天不进行任何操作的情况
            buy = prices[i]
    return max_profit


# nums = [1, 3, 2, 8, 4, 9]
# print(max_profit_6_2(nums, 2))

"""
常见的回溯问题可以分为以下几类：
（1）组合问题：从一组数中按一定规则找出K个数的组合（不考虑顺序）
（2）切割问题：一个字符串按一定规则有几种切割方式
（3）子集问题：一个N个数的集合里有多少符合条件的子集合
（4）排列问题：N个数按一定规则全排列，有几种排列方式（考虑顺序）
（5）棋盘问题：如N皇后、解数独等问题
"""

"""
回溯问题求解模板！

def problem():
    path = []  # 存储遍历过程中满足条件的子结果
    res = []  # 存储最终输出的结果

    def backtrack(当前状态参数):  # 根据问题定义
        if 终止条件:  # 找到一个符合条件的解或者遍历完所有分支
            将满足条件的子结果path添加入最终结果res中
            return
        if 剪枝策略（可选）：
            continue
        for 选择 in 可选择的范围:  # 难点！此处通常决定每次遍历的对象，如字符串的起始、结束位置决定的某个子串（视具体情况而定）
            if 剪枝策略（可选）：  # 提前判断当前选择是否会导致无效路径
                path.append(选择)  # 将遍历过程中满足条件的选择加入到子结果中
                backtrack()  # 递归调用，进入下一层
                path.pop()  # 回溯，撤销当前选择，尝试其它选择

    backtrack(初始状态)
    return res  # 返回最终结果
"""


class CombinationProblem:
    """ 组合问题 """
    @staticmethod
    def combine(n, k):
        """ 77.组合 """
        res = []
        path = []

        def backtrack(start_index, n, k):
            # 终止条件：
            if len(path) == k:
                res.append(path[:])
                print(f'得到一个组合->path: {path}, res: {res}')
                print('-' * 100)
                return
            # 优化：当剩余元素个数不足我们需要的元素时停止
            # (start_index, n - (k - len(path)) + 2)
            # k - len(path)：我们还需要的元素数量
            # 遍历到下标i时列表中剩余的元素为：n+1-i
            # 想要继续遍历，则必须使n+1-i>= k - len(path) ——> i <= n - (k - len(path))+1
            # 则函数遍历的结束位置要写为n - (k - len(path)) + 2
            for i in range(start_index, n - (k - len(path)) + 2):
                path.append(i)
                print(f'递归过程->path: {path}')
                backtrack(i + 1, n, k)
                path.pop()
                print(f'回溯过程->path: {path}')

        backtrack(1, n, k)
        print('-' * 100)
        print(f'输出结果res：{res}')
        return res

    @staticmethod
    def combination_sum3(k, n):
        """ 216. 组合总和III """
        res = []
        path = []

        def backtrack(start_index, k, n, sum_path):
            if sum_path > n:
                print(f'当前总和 {sum_path} 大于目标总和 {n} ->path: {path}')
                print('-' * 50)
                return
            if len(path) == k:
                if sum_path == n:
                    res.append(path[:])
                    print(f'得到一个组合->path: {path}, res: {res}')
                    print('-' * 50)
                    return
            for i in range(start_index, 9 - (k - len(path)) + 2):
                path.append(i)
                print(f'递归过程->path: {path}')
                backtrack(i + 1, k, n, sum_path + i)  # 将求和作为递归参数之一
                path.pop()
                print(f'回溯过程->path: {path}')

        backtrack(1, k, n, 0)
        print('-' * 50)
        print(f'输出结果res：{res}')
        return res


CP = CombinationProblem()
# print(CP.combine(4, 2))
print(CP.combination_sum3(3, 7))
"""
示例 1：

输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
"""


class PartitionProblem:
    """ 分割问题 """
    @staticmethod
    def restore_ip_addresses(s):
        """ 93.复原IP地址 """
        path = []
        res = []

        def backtrack(s, start_index):
            # 终止条件1：如果给定得字符串长度超过12直接返回空，因为有效IP最多12个数字
            if len(s) > 12:
                return []
            # 终止条件2：中间结果的长度为4说明此时已经得到了4个可以组成有效IP的数字
            if len(path) == 4:
                # 终止条件3：起始下标位于字符串末尾位置，说明已经将整个字符串都遍历结束了
                # 终止条件2，3需同时满足才能说明将已知的整个字符串划分为了能够组成有效IP的4个数字
                if start_index == len(s):
                    res.append('.'.join(path[:]))  # 将path中的数字以'.'连接起来
                    print(f"得到有效中间结果path：{['.'.join(path[:])]}, 当前结果res：{res}")
                    print('--' * 45)
                    print('达到终止条件，将path加入到res后返回上一级，开始回溯')
                    return
            if start_index == len(s):
                print('--' * 45)
                print(f'分割起始索引 {start_index} 已超出字符串最大索引 {len(s)-1}，需返回上一级，开始回溯')
            for i in range(start_index, min(start_index + 3, len(s))):
                # start_index代表分割的起始位置，要确保已经被分割的部分不被重复分割
                # i代表分割的结束位置，因为一个有效数字最多是三位数，
                # 所以起始位置和结束位置最多差3，或者是遍历到达字符串s的末尾位置，两种情况取最小值。
                t = s[start_index: i + 1]  # 目前被分割到的子串
                # 剪枝条件1：如果分割得到的数字小于0或者大于255则无效，直接跳出当前循环
                if not 0 <= int(t) <= 255:
                    print(f'分割起始、结束索引：{[start_index, i]}, 无效子串(值的大小超出范围)：{s[start_index: i + 1]},'
                          f' 当前path：{path}, 继续往后分割直至结束索引达到最大索引时回溯')
                    continue
                # 剪枝条件2：如果分割得到的数字不等于0 且 在去除左侧的数字0之后与其不相等，
                # 说明分割得到的这个数字是以0开头的，即含义前导0，于题意不符，直接跳出当前循环
                elif t != '0' and t.lstrip('0') != t:
                    print(f'分割起始、结束索引：{[start_index, i]}, 无效子串(含先导0)：{s[start_index: i + 1]},'
                          f' 当前path：{path}, 继续往后分割直至结束索引达到最大索引时回溯')
                    continue
                else:
                    # 满足上述诸多条件之后，则说明是一个合格的数字子串，可以作为有效IP的其中一个组成部分加入到path中
                    path.append(t)
                    print(f'分割起始、结束索引：{[start_index, i]}, 递归后的起始索引：{i + 1}, 有效子串：{s[start_index: i + 1]}, 当前path：{path}')
                    backtrack(s, i + 1)  # 递归，去掉已经分割的部分字串，下一次分割的起始位置便是上一次的结束位置+1
                    path.pop()  # 回溯
                    print(f'回溯到起始索引为 {start_index} 的位置，回溯后的当前path：{path}')

        print(f'输入字符串：{s}, 输入字符串长度：{len(s)}')
        print('--' * 45)
        backtrack(s, 0)
        return res

    @staticmethod
    def partition(s):
        """ 131.分割回文串 """
        path = []
        res = []

        def is_palindrome(t):
            """ 是否回文子串 """
            if len(t) <= 1:
                return True
            i, j = 0, len(t) - 1
            while i <= j:
                if t[i] != t[j]:  # 从两头向中间遍历
                    return False
                i += 1
                j -= 1
            return True

        def backtrack(s, start_index):
            # 终止条件：起始位置等于字符串长度
            # 这里不是len(s)-1的原因是最后一步执行递归的时候传入的数字（i+1）肯定等于len(s)
            if start_index == len(s):
                res.append(path[:])
                print(f"得到有效中间结果path：{['.'.join(path[:])]}, 当前结果res：{res}")
                print('--' * 45)
                print('达到终止条件，将path加入到res后返回上一级，开始回溯')
                return
            for i in range(start_index, len(s)):
                # start_index代表分割的起始位置，因为要确保已经分割过的部分不能再分割
                # 遍历过程中的i代表分割的结束位置
                # 当目前选择的字串是回文子串时执行下述递归回溯操作
                if is_palindrome(s[start_index: i + 1]):
                    path.append(s[start_index: i + 1])
                    print(f'分割起始、结束索引：{[start_index, i]}, 递归后的起始索引：{i + 1}, 回文子串：{s[start_index: i + 1]}, 当前path：{path}')
                    # 因为字符串的[start_index: i + 1]部分已经作为一个回文字串被加入到path中了
                    # 那么接下来分割的起始位置就应该是i + 1了
                    backtrack(s, i + 1)
                    path.pop()  # 回溯操作
                    print(f'回溯到起始索引为 {start_index} 的位置，回溯后的当前path：{path}')

                else:
                    print(f'分割起始、结束索引：{[start_index, i]}, 非回文子串：{s[start_index: i + 1]},'
                          f' 当前path：{path}, 继续往后分割直至结束索引达到最大索引时回溯')

        print(f'输入字符串：{s}, 输入字符串长度：{len(s)}')
        print('--' * 45)
        backtrack(s, 0)
        return res


PP = PartitionProblem()
# c = "aab"
# print(f'最终输出结果res：{PP.partition(c)}')
# c = "11023"
# print(f'最终输出结果res：{PP.restore_ip_addresses(c)}')


class PermuteProblem:
    """
    排列问题: 每次都是从头开始搜索，不需要start_index参数
    """
    @staticmethod
    def permute(nums):
        """ 46.全排列 """
        path = []
        res = []
        num_used = [False] * len(nums)  # 记录每个位置元素的使用情况

        def backtrack(nums):
            # 终止条件：每一个全排列的长度肯定都是等于数组长度的
            if len(path) == len(nums):
                res.append(path[:])
                print(f'得到一个全排列->path: {path}, res: {res}')
                print('-' * 100)
                return
            # 每次都从头开始遍历
            for i in range(len(nums)):
                # 确保对应位置的元素未被使用过，否则元素会被重复使用，使最终结果中出现重复的全排列
                if not num_used[i]:
                    num_used[i] = True  # 标记该元素已被使用
                    path.append(nums[i])
                    print(f'递归过程->path: {path}, num_used: {num_used}')
                    backtrack(nums)  # 递归
                    path.pop()  # 回溯
                    num_used[i] = False  # 撤回该元素的使用
                    print(f'回溯过程->path: {path}, num_used: {num_used}')

        print(f'输入数组nums：{nums}')
        print('-' * 100)
        backtrack(nums)
        print('-' * 100)
        print(f'输出结果res：{res}')
        return res

    @staticmethod
    def permute_unique(nums):
        """ 47.全排列Ⅱ """
        path = []
        res = []
        num_used = [False] * len(nums)  # 记录每个位置元素的使用情况

        def backtrack(nums):
            # 终止条件：每一个全排列的长度肯定都是等于数组长度的
            if len(path) == len(nums):
                res.append(path[:])
                print(f'得到一个全排列->path: {path}, res: {res}')
                print('-' * 100)
                return
            for i in range(len(nums)):
                # 剪枝条件1：如果该元素已经使用过，则跳过
                if num_used[i]:
                    continue
                # 剪枝条件2：如果该元素与前一个元素相等且前一个元素使用过，则跳过
                if i > 0 and nums[i] == nums[i - 1] and num_used[i - 1]:
                    continue
                num_used[i] = True  # 标记该元素已被使用
                path.append(nums[i])
                print(f'递归过程->path: {path}, num_used: {num_used}')
                backtrack(nums)  # 递归
                path.pop()  # 回溯
                num_used[i] = False  # 撤回该元素的使用
                print(f'回溯过程->path: {path}, num_used: {num_used}')

        # 必须先排序：因为输入当中含有重复元素，需要确保在一个全排列中不能有重复元素，
        # 那么使重复元素相邻排列相对于最后一步对结果去重来说是最易于实现的。
        nums.sort()
        print(f'排序后的输入数组nums：{nums}')
        print('-' * 100)
        backtrack(nums)
        print('-' * 100)
        print(f'输出结果res：{res}')
        return res


PEP = PermuteProblem()
# lists = [1, 2, 3]
# print(PEP.permute(lists))
# print(PEP.permute_unique(lists))

"""
示例
输入：nums = [1,1,2]
输出：[[1,1,2], [1,2,1], [2,1,1]]
"""


class SubsetProblem:
    """ 子集问题 """
    @staticmethod
    def subsets(nums):
        """ 78.子集 """
        path = []
        res = []

        def backtrack(nums, start_index):
            res.append(path[:])
            print(f'得到一个子集->path: {path}, res: {res}')
            print('-' * 100)
            for i in range(start_index, len(nums)):
                path.append(nums[i])
                print(f'递归过程->path: {path}')
                backtrack(nums, i + 1)
                path.pop()
                print(f'回溯过程->path: {path}')

        print(f'输入数组nums：{nums}')
        print('-' * 100)
        backtrack(nums, 0)
        print('-' * 100)
        print(f'输出结果res：{res}')
        return res

    @staticmethod
    def subsets_unique(nums):
        """ 90.子集Ⅱ """
        path = []
        res = []

        def backtrack(nums, start_index):
            res.append(path[:])
            print(f'得到一个子集->path: {path}, res: {res}')
            print('-' * 100)
            for i in range(start_index, len(nums)):
                if i > start_index and nums[i] == nums[i - 1]:
                    print(f'遇到重复元素{nums[i]}->path: {path}, res: {res}')
                    print('-' * 100)
                    continue
                path.append(nums[i])
                print(f'递归过程->path: {path}')
                backtrack(nums, i + 1)
                path.pop()
                print(f'回溯过程->path: {path}')

        print(f'输入数组nums：{nums}')
        print('-' * 100)
        nums.sort()
        print(f'排序后的输入数组nums：{nums}')
        print('-' * 100)
        backtrack(nums, 0)
        print('-' * 100)
        print(f'输出结果res：{res}')
        return res

    @staticmethod
    def find_sub_sequences(nums):
        """ 491.非递减子序列 """
        path = []
        res = []

        def backtrack(nums, start_index):
            if len(path) >= 2:
                res.append(path[:])
            if start_index == len(nums):
                return
            num_used = set()
            for i in range(start_index, len(nums)):
                if (path and nums[i] < path[-1]) or nums[i] in num_used:
                    continue
                num_used.add(nums[i])
                path.append(nums[i])
                backtrack(nums, i + 1)
                path.pop()

        backtrack(nums, 0)
        return res


SP = SubsetProblem()
# lists = [1, 3, 1]
# print(SP.subsets(lists))
# print(SP.subsets_unique(lists))
# print(SP.find_sub_sequences(lists))

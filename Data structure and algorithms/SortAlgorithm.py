class SortAlgorithm:

    def merge_sort(self, nums):
        """
        归并排序：递归地将数组切分到最小单位，再逆向逐步合并
        时间复杂度：O(nlogn),因为每次分割数组的复杂度为O(logn),合并的复杂度为O(n)
        空间复杂度：O(n),用于储存结果
        """
        n = len(nums)
        if n <= 1:
            return nums

        mid = n // 2
        left = nums[:mid]
        right = nums[mid:]

        left = self.merge_sort(left)
        right = self.merge_sort(right)

        res = [0] * n
        i, j, k = 0, 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res[k] = left[i]
                i += 1
            else:
                res[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            res[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            res[k] = right[j]
            j += 1
            k += 1

        return res

    def quick_sort(self, nums):
        """
        快速排序：每次都要选择一个基准，将数组分为两部分，一部分都比基准小，一部分都比基准大
        时间复杂度：平均情况:O(nlogn),最坏情况(选取的基准总是最小或最大元素):O(n^2),最好情况：O(nlogn)
        空间复杂度：O(nlogn),递归调用栈的空间
        """
        return self.quick_sort_helper(nums, 0, len(nums) - 1)

    def quick_sort_helper(self, nums, first, last):
        """ 快速排序辅助函数 """
        if first < last:
            pivot = nums[first]

            i, j = first + 1, last
            while True:
                while i <= j and nums[i] <= pivot:
                    i += 1
                while i <= j and nums[i] >= pivot:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                else:
                    break
            nums[first], nums[j] = nums[j], nums[first]

            self.quick_sort_helper(nums, first, j - 1)
            self.quick_sort_helper(nums, j + 1, last)
        return nums

    @staticmethod
    def bubble_sort(nums):
        """
        冒泡排序：每一轮都交换相邻位置的两个数字,较大的数字位置靠后
        时间复杂度：平均情况：O(n^2),最坏情况(每个元素都要比较n次):O(n^2),最好情况(即数组有序):O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        exchange = True
        i = 0
        while i < n and exchange:  # 如果某一轮遍历过程当中没有发生交换操作则说明数组此时已经完成排序
            exchange = False
            for j in range(n-i-1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    exchange = True
            i += 1
        return nums

    @staticmethod
    def select_sort(nums):
        """
        选择排序：每次都选择一个最大的数字放在最后一个位置上
        时间复杂度：O(n^2),无论如何每次都要扫描剩余元素从中选出最大值。
        空间复杂度：O(1)
        """
        n = len(nums)
        for i in range(n - 1, 0, -1):
            max_pos = 0
            for j in range(1, i + 1):
                if nums[j] > nums[max_pos]:
                    max_pos = j
            nums[i], nums[max_pos] = nums[max_pos], nums[i]
        return nums

    @staticmethod
    def insert_sort(nums):
        """
        插入排序：从第二个数字开始从后往前寻找每一个数字应该插入的正确位置
        时间复杂度：平均情况：O(n^2),最坏情况(数组完全逆序)：O(n^2),最好情况(数组有序)：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        for i in range(1, n):
            current_value = nums[i]
            pos = i
            while pos > 0 and nums[pos - 1] > current_value:
                nums[pos] = nums[pos - 1]
                pos -= 1
            nums[pos] = current_value
        return nums

    def heapify(self, nums, n, i):
        """ 堆排序辅助函数：根据数组、数组长度和当前索引构建大根堆 """
        root = i
        left = 2 * i + 1
        right = 2 * i + 2
        # 如果左子节点值大于根节点值
        if left < n and nums[left] > nums[root]:
            root = left
        # 如果右子节点值大于跟节点值
        if right < n and nums[right] > nums[root]:
            root = right

        if root != i:
            nums[i], nums[root] = nums[root], nums[i]
            # 递归地调整后续受影响的子树
            self.heapify(nums, n, root)

    def heap_sort(self, nums):
        """
        堆排序：利用堆这种数据结构进行排序，构建大根堆
        时间复杂度：O(nlogn)
        空间复杂度：O(1)
        """
        n = len(nums)
        # 构建大根堆：从最后一个非叶子节点的索引开始，也就是n//2-1
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(nums, n, i)
        # 从根节点开始逐个提取最大值，移动到末尾
        for i in range(n - 1, 0, -1):
            nums[i], nums[0] = nums[0], nums[i]
            self.heapify(nums, i, 0)

        return nums

    @staticmethod
    def bucket_sort(nums):
        """
        桶排序：将数据分到多个桶中进行排序再合并
        时间复杂度：O(n+k),k是桶的数量
        空间复杂度：O(n+k)
        """
        if not nums:
            return nums
        n = len(nums)
        max_value, min_value = max(nums), min(nums)
        bucket_size = (max_value - min_value) // n + 1  # 计算桶的数量
        buckets = [[] for _ in range(bucket_size)]

        # 将每个元素添加到对应的桶中
        for num in nums:
            index = (num - min_value) // n
            buckets[index].append(num)

        # 在桶内排序，后合并
        sort_buckets = []
        for bucket in buckets:
            sort_buckets.extend(sorted(bucket))  # 可以用其它排序算法替换

        return sort_buckets

    @staticmethod
    def shell_sort(nums):
        """
        希尔排序:基于插入排序，将数组分为若干个子序列，逐渐减小间隔直至最终进行一次插入排序
        时间复杂度：依赖于增量序列，通常在O(n^1.3)-O(n^2)之间
        空间复杂度：O(1)
        """
        n = len(nums)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = nums[i]
                j = i
                while j >= gap and nums[j - gap] > temp:
                    nums[j] = nums[j - gap]
                    j -= gap
                nums[j] = temp
            gap //= 2

        return nums

    @staticmethod
    def counting_sort(nums):
        """
        计数排序：计数元素出现的次数并将数据重组
        时间复杂度：O(n+k),k为数据范围的长度
        空间复杂度：O(n+k)
        """
        n = len(nums)
        max_value, min_value = max(nums), min(nums)
        range_value = max_value - min_value + 1  # 数据范围的长度

        count = [0] * range_value  # 计数数组，存储每个元素出现的次数
        output = [0] * n  # 输出数组，存储排序后的结果

        # 统计元素出现的次数，num - min_value是为了将所有数值偏移到从0开始的位置
        for num in nums:
            count[num - min_value] += 1

        # 将计数数组更新为累加数组，每个位置的值代表当前元素及之前所有元素的累计出现次数
        for i in range(1, len(count)):
            count[i] += count[i - 1]

        # 将元素放置到输出数组，倒序遍历数组元素
        for num in reversed(nums):
            output[count[num - min_value] - 1] = num  # 确定num在output数组中的位置
            count[num - min_value] -= 1  # 放入一个元素，对应次数减一

        return output

    @staticmethod
    def radix_sort_helper(nums, exp):
        """ 基数排序辅助函数 """
        n = len(nums)
        output = [0] * n
        count = [0] * 10

        # 统计每个数字在当前位的出现次数
        for num in nums:
            index = (num // exp) % 10
            count[index] += 1

        # 将计数数组转换为累加数组
        for i in range(1, 10):
            count[i] += count[i - 1]

        # 将元素放入输出数组，倒序遍历确保稳定性
        for i in range(n - 1, -1, -1):
            index = (nums[i] // exp) % 10
            output[count[index] - 1] = nums[i]
            count[index] -= 1

        for i in range(n):
            nums[i] = output[i]

    def radix_sort(self, nums):
        """
        基数排序：从最低位开始，依次对每一位进行排序，
        时间复杂度：O(nd),d为位数
        空间复杂度：O(n+k),k为位数范围
        """
        max_value = max(nums)  # 找到最大值以确定最大位数
        exp = 1  # 初始的位数，从个位开始

        # 依次对每个位进行排序，直至所有位都处理完
        while max_value // exp > 0:
            self.radix_sort_helper(nums, exp)
            exp *= 10  # 增加位数，进行下一位的排序

        return nums


lists = [1, 4, 22, 7, 15, 6, 33, 8]
sort = SortAlgorithm()
# print(sort.merge_sort(lists))
# print(sort.quick_sort(lists))
# print(sort.bubble_sort(lists))
# print(sort.select_sort(lists))
# print(sort.insert_sort(lists))
# print(sort.heap_sort(lists))
# print(sort.bucket_sort(lists))
# print(sort.shell_sort(lists))
# print(sort.counting_sort(lists))
print(sort.radix_sort(lists))









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


# lists = [1, 4, 22, 7, 15, 6, 33, 8]
# sort = SortAlgorithm()
# print(sort.merge_sort(lists))
# print(sort.quick_sort(lists))
# print(sort.bubble_sort(lists))
# print(sort.select_sort(lists))
# print(sort.insert_sort(lists))
# print(sort.heap_sort(lists))
# print(sort.bucket_sort(lists))
# print(sort.shell_sort(lists))
# print(sort.counting_sort(lists))
# print(sort.radix_sort(lists))


def quick_sort(nums):
    """
    快速排序
    时间复杂度：1、理想情况下，每个选择的分割值恰好将数组均分为两个大小相等的子数组时，
                  递归的深度为log(n)，因此递归的时间复杂度为0(logn)
                  而每层递归都需要遍历整个数组，时间复杂度为0(n)
                  总的时间复杂度为0(nlogn)
              2、平均情况下，分隔值的划分大概率接近平衡，因此平均时间复杂度仍为0(nlogn)
              3、最坏情况下，当输入数组已经是有序的，且每次选择第一个元素作为分割值时进行划分时，
                  会导致极为不平衡的子数组，即一个大小为0，一个大小为n，
                  此时递归深度为n，因此递归的时间复杂度为0(n)。
                  而每层递归都需要遍历整个数组，时间复杂度为0(n)
                  总的时间复杂度为0(n^2)
    空间复杂度：快速排序是原地排序，非递归部分的空间复杂度为O(1)，因此只考虑递归的空间复杂度
              1、最优/平均情况：递归深度为logn，空间复杂度为O(logn)。
              2、最坏情况：递归深度为n，空间复杂度退化为O(n)
    """
    return quick_sort_helper(nums, 0, len(nums) - 1)


def quick_sort_helper(nums, left, right):
    """ 对数组递归地进行快速排序 """
    if left < right:
        pivot = partition(nums, left, right)  # 找到划分数组的分割点
        quick_sort_helper(nums, left, pivot - 1)  # 对分割点左侧数组递归排序
        quick_sort_helper(nums, pivot + 1, right)  # 对分割点右侧数组递归排序
    return nums


def partition(nums, left, right):
    """为选定的分割值找到一个合适的位置，使将数组中小于分割值的元素置于分割值左侧，大于分割值的元素置于分割值右侧 """
    pivot_val = nums[left]  # 选择数组的第一个元素为分割值
    i, j = left + 1, right  # 双指针，从两边向中间遍历
    while True:
        # 若i<=j且当前i指针指向的元素小于等于分割值，则i向右移动一位
        while i <= j and nums[i] <= pivot_val:
            i += 1
        # 若i<=j且当前j指针指向的元素大于等于分割值，则j向左移动一位
        while i <= j and nums[j] >= pivot_val:
            j -= 1
        # 若i<=j且当前i指针指向的元素大于等于分割值，j指针指向的元素小于等于分割值
        # 此时两个指针都无法移动，则将两个指针指向的元素进行交换
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        # 遍历结束，推出循环
        else:
            break
    # 最终将分割值移动到合适的分割位置上，即位置j(遍历结束时，j指针所指的元素是小于基准值的元素)
    nums[left], nums[j] = nums[j], nums[left]
    return j


# lists = [5, 4, 22, 7, 15, 6, 33, 8]
# print(quick_sort(lists))


def merge_sort(nums):
    """
    归并排序
    时间复杂度：分解阶段，将数组递归地二分，直到子数组长度为1。分解次数即递归深度，为logn，时间复杂度为O(logn)
              合并阶段，每层递归地合并两个子数组，合并操作的比较次数和赋值次数与当前子数组的总长度成正比，即每层合并的时间复杂度为O(n)
              总的时间复杂度为0(nlogn)
    空间复杂度：递归过程中栈的调用带来的空间复杂度为O(logn)
              合并过程中维护了一个大小为n的数组储存最终结果，空间复杂度为O(n)
              合并操作中的数组占主导地位，因此总的空间复杂度为O(n)
    """
    n = len(nums)
    if n <= 1:
        return nums
    # 递归分解阶段
    mid = n // 2
    left = nums[:mid]
    right = nums[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    # 合并阶段
    res = [0] * n
    i, j, k = 0, 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res[k] = left[i]
            k += 1
            i += 1
        else:
            res[k] = right[j]
            k += 1
            j += 1
    while i < len(left):
        res[k] = left[i]
        k += 1
        i += 1
    while j < len(right):
        res[k] = right[j]
        k += 1
        j += 1
    return res


lists = [5, 4, 22, 7, 15, 6, 33, 8]
# print(merge_sort(lists))


def quick_sort_1(nums):
    return quick_sort_helper(nums, 0, len(nums) - 1)


def quick_sort_helper_1(nums, first, last):
    if first < last:
        pivot = partition(nums, first, last)
        quick_sort_helper_1(nums[:pivot], first, pivot - 1)
        quick_sort_helper_1(nums[pivot+1:], pivot + 1, last)


def partition_1(nums, first, last):
    pivot_val = nums[first]
    left, right = first + 1, last
    while True:
        while left <= right and nums[left] < pivot_val:
            left += 1
        while left <= right and nums[right] > pivot_val:
            right -= 1
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        else:
            break
    nums[first], nums[left] = nums[left], nums[first]
    return left


print(quick_sort_1(lists))





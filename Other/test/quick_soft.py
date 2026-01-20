def quick_sort(arr):
    """
    快速排序主函数
    :param arr: 待排序的列表
    :return: 排序后的列表
    """
    # 如果数组长度小于等于1，直接返回（递归终止条件）
    if len(arr) <= 1:
        return arr

    # 选择基准元素（这里选择中间元素，避免最坏情况）
    pivot = arr[len(arr) // 2]

    # 将数组分成三部分：小于基准、等于基准、大于基准
    left = [x for x in arr if x < pivot]  # 小于基准的元素
    middle = [x for x in arr if x == pivot]  # 等于基准的元素
    right = [x for x in arr if x > pivot]  # 大于基准的元素

    # 递归地对左右两部分排序，然后合并结果
    return quick_sort(left) + middle + quick_sort(right)


def quick_sort_inplace(arr, low=0, high=None):
    """
    原地快速排序（不创建新列表，节省内存）
    :param arr: 待排序的列表
    :param low: 起始索引
    :param high: 结束索引
    """
    if high is None:
        high = len(arr) - 1

    if low < high:
        # 获取分区点索引
        pi = partition(arr, low, high)

        # 递归地对分区点左右两部分排序
        quick_sort_inplace(arr, low, pi - 1)  # 排序左半部分
        quick_sort_inplace(arr, pi + 1, high)  # 排序右半部分


def partition(arr, low, high):
    """
    分区函数：将数组分区，返回基准元素的正确位置索引
    """
    # 选择最右边的元素作为基准
    pivot = arr[high]

    # i 指向小于基准的区域的边界
    i = low - 1

    # 遍历数组（从 low 到 high-1）
    for j in range(low, high):
        # 如果当前元素小于等于基准
        if arr[j] <= pivot:
            i += 1
            # 将小于基准的元素交换到左边
            arr[i], arr[j] = arr[j], arr[i]

    # 将基准元素放到正确的位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    # 返回基准元素的最终位置
    return i + 1


# 测试快速排序
def test_quick_sort():
    # 测试数据
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1],
        [],
        [3, 3, 3, 3],
        [10, 7, 8, 9, 1, 5]
    ]

    print("快速排序测试:")
    print("=" * 50)

    for i, arr in enumerate(test_cases):
        # 创建副本用于原地排序测试
        arr_copy = arr.copy()

        # 使用非原地版本排序
        sorted_arr = quick_sort(arr)

        # 使用原地版本排序
        quick_sort_inplace(arr_copy)

        print(f"测试用例 {i + 1}:")
        print(f"原数组: {arr}")
        print(f"非原地排序结果: {sorted_arr}")
        print(f"原地排序结果: {arr_copy}")
        print()


# 可视化快速排序过程
def visualize_quick_sort(arr, depth=0, side=""):
    """
    可视化快速排序的递归过程
    """
    if len(arr) <= 1:
        print(f"{'  ' * depth}{side}叶子节点: {arr}")
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    print(f"{'  ' * depth}{side}数组: {arr}")
    print(f"{'  ' * depth}基准: {pivot}")
    print(f"{'  ' * depth}左分区: {left}")
    print(f"{'  ' * depth}中分区: {middle}")
    print(f"{'  ' * depth}右分区: {right}")
    print(f"{'  ' * depth}{'-' * 30}")

    # 递归处理左右分区
    sorted_left = visualize_quick_sort(left, depth + 1, "左: ")
    sorted_right = visualize_quick_sort(right, depth + 1, "右: ")

    result = sorted_left + middle + sorted_right

    print(f"{'  ' * depth}{side}合并结果: {result}")
    return result


if __name__ == "__main__":
    # 运行测试
    test_quick_sort()

    print("\n\n快速排序过程可视化:")
    print("=" * 50)
    test_array = [33, 10, 15, 7, 20, 15, 12]
    print(f"原始数组: {test_array}")
    print("-" * 50)
    visualize_quick_sort(test_array)
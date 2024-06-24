import numpy as np


class EnhancedIPVO:
    def __init__(self, image, secret_data, t1, t2):
        """
        初始化EPVO-RDH方案的参数。

        参数:
        image (ndarray): 输入的灰度图像。
        secret_data (list): 要嵌入的比特数据列表。
        t1 (float): 平滑阈值。
        t2 (float): 复杂阈值。
        """
        self.image = image  # 保存输入图像
        self.secret_data = secret_data  # 保存秘密数据
        self.t1 = t1  # 平滑阈值
        self.t2 = t2  # 复杂阈值
        self.height, self.width = image.shape  # 获取图像的高度和宽度
        self.stego_image = np.copy(image)  # 创建图像的副本用于存储嵌入数据后的图像
        self.data_index = 0  # 用于跟踪嵌入数据的位置

    def get_rhombus_mean(self, i, j):
        """
        计算(i, j)位置像素的菱形均值。

        参数:
        i (int): 行索引。
        j (int): 列索引。

        返回:
        float: 菱形均值。
        """
        neighbors = []  # 存储邻居像素
        if i > 0:
            neighbors.append(self.image[i - 1, j])  # 上方像素
        if i < self.height - 1:
            neighbors.append(self.image[i + 1, j])  # 下方像素
        if j > 0:
            neighbors.append(self.image[i, j - 1])  # 左方像素
        if j < self.width - 1:
            neighbors.append(self.image[i, j + 1])  # 右方像素
        return np.mean(neighbors)  # 计算邻居像素的均值

    def categorize_block(self, block):
        """
        根据菱形均值对图像块进行分类。

        参数:
        block (list): 图像块中的像素位置。

        返回:
        str: 分类结果（"smooth" 或 "complex"）。
        """
        means = [self.get_rhombus_mean(i, j) for i, j in block]  # 计算每个像素的菱形均值
        mu1 = means[0]  # 第一个像素的均值
        mu2 = means[1]  # 第二个像素的均值
        mu3 = means[2]  # 第三个像素的均值
        # 判断是否属于平滑块
        if (mu1 - self.t1) < mu2 < (mu1 + self.t1) and (mu1 - self.t1) < mu3 < (mu1 + self.t1):
            return "smooth"
        else:
            return "complex"

    def embed_phase1(self, block):
        """
        第一阶段嵌入：使用增强型像素值排序（EPI-PVO）方法。

        参数:
        block (list): 平滑图像块中的像素位置。
        """
        # 获取块内的原始像素值
        original_values = [self.image[i, j] for i, j in block]
        # 对块内的像素值进行排序，保留原始索引
        sorted_indices = sorted(range(len(original_values)), key=lambda x: original_values[x])
        min_index = sorted_indices[0]  # 最小像素值的原始索引
        second_min_index = sorted_indices[1]  # 次小像素值的原始索引
        max_index = sorted_indices[-1]  # 最大像素值的原始索引
        second_max_index = sorted_indices[-2]  # 次大像素值的原始索引

        min_value_pos = block[min_index]
        max_value_pos = block[max_index]

        if min_index < second_min_index:  # 判断最小像素值的原始索引和次小像素值的原始索引谁比较小
            min_pos = block[min_index]  # 把较小的索引的像素块的位置赋给最小像素下标的位置
            second_min_pos = block[second_min_index]  # 把较大的索引的像素块的位置赋给次小像素下标的位置
        else:
            min_pos = block[second_min_index]
            second_min_pos = block[min_index]

        if max_index < second_max_index:
            max_pos = block[second_max_index]
            second_max_pos = block[max_index]
        else:
            max_pos = block[max_index]
            second_max_pos = block[second_max_index]

        E_min = self.image[min_pos[0], min_pos[1]] - self.image[second_min_pos[0], second_min_pos[1]]  # 计算最小误差值
        E_max = self.image[second_max_pos[0], second_max_pos[1]] - self.image[max_pos[0], max_pos[1]]  # 计算最大误差值

        if self.data_index < len(self.secret_data):  # 检查是否还有秘密数据未嵌入
            # 嵌入第一个秘密比特到最小预测误差
            bit = self.secret_data[self.data_index]  # 获取当前要嵌入的秘密数据比特

            if E_min in [0, 1]:
                self.stego_image[min_value_pos[0], min_value_pos[1]] = max(
                    self.image[min_value_pos[0], min_value_pos[1]] - bit, 0)  # 减去秘密比特值
                self.data_index += 1  # 嵌入成功后移动到下一个要嵌入的比特
            else:
                self.stego_image[min_value_pos[0], min_value_pos[1]] = max(
                    self.image[min_value_pos[0], min_value_pos[1]] - 1, 0)  # 减1

            if self.data_index < len(self.secret_data):  # 检查是否还有秘密数据未嵌入
                # 嵌入当前秘密比特到最大预测误差
                bit = self.secret_data[self.data_index]  # 获取当前要嵌入的秘密数据比特

                if E_max in [0, 1]:
                    self.stego_image[max_value_pos[0], max_value_pos[1]] = min(
                        self.image[max_value_pos[0], max_value_pos[1]] + bit, 255)  # 加上秘密比特值
                    self.data_index += 1  # 移动到下一个要嵌入的比特
                else:
                    self.stego_image[max_value_pos[0], max_value_pos[1]] = min(
                        self.image[max_value_pos[0], max_value_pos[1]] + 1, 255)  # 加1

    def embed_phase2(self, block):
        """
        第二阶段嵌入：使用基于恢复的预测误差嵌入方法。

        参数:
        block (list): 平滑图像块中的像素位置。
        """
        sorted_block = sorted(block, key=lambda x: self.get_rhombus_mean(x[0], x[1]))  # 对块内像素的菱形均值进行排序
        min_pos = sorted_block[0]  # 最小值像素位置
        mid_pos = sorted_block[1]  # 中间值像素位置
        max_pos = sorted_block[2]  # 最大值像素位置

        if self.data_index < len(self.secret_data):  # 检查是否还有秘密数据未嵌入
            bit = self.secret_data[self.data_index]  # 获取当前要嵌入的秘密数据比特
            E1 = self.image[min_pos[0], min_pos[1]] - self.get_rhombus_mean(min_pos[0], min_pos[1])  # 计算最小值像素的预测误差
            E3 = self.image[max_pos[0], max_pos[1]] - self.get_rhombus_mean(max_pos[0], max_pos[1])  # 计算最大值像素的预测误差

            if bit == 1:
                self.stego_image[min_pos[0], min_pos[1]] = min(self.image[min_pos[0], min_pos[1]] + 1,
                                                               255)  # 如果比特为1，最小值像素加1
                self.stego_image[max_pos[0], max_pos[1]] = max(self.image[max_pos[0], max_pos[1]] - 1,
                                                               0)  # 如果比特为1，最大值像素减1
            else:
                self.stego_image[min_pos[0], min_pos[1]] = self.image[min_pos[0], min_pos[1]]  # 如果比特为0，保持最小值像素不变
                self.stego_image[max_pos[0], max_pos[1]] = self.image[max_pos[0], max_pos[1]]  # 如果比特为0，保持最大值像素不变

            E2 = self.image[mid_pos[0], mid_pos[1]] - self.get_rhombus_mean(mid_pos[0], mid_pos[1])  # 计算中间值像素的预测误差
            if E2 in [0, 1]:
                if bit == 1:
                    self.stego_image[mid_pos[0], mid_pos[1]] = min(self.image[mid_pos[0], mid_pos[1]] + 1,
                                                                   255)  # 如果预测误差为0或1且比特为1，则加1
                else:
                    self.stego_image[mid_pos[0], mid_pos[1]] = max(self.image[mid_pos[0], mid_pos[1]] - 1,
                                                                   0)  # 如果预测误差为0或1且比特为0，则减1
            else:
                if bit == 1:
                    self.stego_image[mid_pos[0], mid_pos[1]] = self.image[mid_pos[0], mid_pos[1]] + 1  # 如果比特为1，中间值像素加1
                else:
                    self.stego_image[mid_pos[0], mid_pos[1]] = self.image[mid_pos[0], mid_pos[1]] - 1  # 如果比特为0，中间值像素减1

            self.data_index += 1  # 移动到下一个要嵌入的比特

    def embed_data_in_smooth_block(self, block):
        """
        在平滑块中嵌入数据。

        参数:
        block (list): 平滑图像块中的像素位置。
        """
        self.embed_phase1(block)  # 执行第一阶段嵌入
        self.embed_phase2(block)  # 执行第二阶段嵌入

    def embed_data_in_complex_block(self, block):
        """
        在复杂块中嵌入数据。

        参数:
        block (list): 复杂图像块中的像素位置。
        """
        for i, j in block:  # 遍历块内的每个像素
            if self.data_index < len(self.secret_data):  # 检查是否还有秘密数据未嵌入
                bit = self.secret_data[self.data_index]  # 获取当前要嵌入的秘密数据比特
                if bit == 1:
                    self.stego_image[i, j] = min(self.image[i, j] + 1, 255)  # 如果比特为1，像素加1
                else:
                    self.stego_image[i, j] = max(self.image[i, j] - 1, 0)  # 如果比特为0，像素减1
                self.data_index += 1  # 移动到下一个要嵌入的比特

    def embed_data(self):
        """
        嵌入秘密数据到图像中。
        """
        block_size = 3  # 设置块大小
        # 遍历图像的每个3x1块
        for k in range(0, self.height, block_size):  # 按蛇形遍历图像的行
            if k // block_size % 2 == 0:  # 偶数行从左到右
                cols = range(self.width)
            else:  # 奇数行从右到左
                cols = range(self.width - 1, -1, -1)
            for j in cols:
                block = [(k + i, j) for i in range(block_size) if k + i < self.height]  # 获取块内的像素位置
                if len(block) == block_size:  # 确保块的大小为3x1
                    block_type = self.categorize_block(block)  # 分类块
                    if block_type == "smooth":
                        self.embed_data_in_smooth_block(block)  # 如果是平滑块，进行平滑块嵌入
                    else:
                        self.embed_data_in_complex_block(block)  # 如果是复杂块，进行复杂块嵌入


# 示例用法
# 假设我们有一个灰度图像（numpy数组）和一些秘密数据
image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)  # 生成一个512x512的随机灰度图像
secret_data = [1, 0, 1, 1, 0] * 10000  # 生成一些示例秘密数据

# 创建算法实例并嵌入数据
epivo = EnhancedIPVO(image, secret_data, t1=0.1, t2=0.5)  # 创建EPIVO实例
epivo.embed_data()  # 嵌入数据

# 获取嵌入数据后的图像
stego_image = epivo.stego_image  # 获取嵌入数据后的图像

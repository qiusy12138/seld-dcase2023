# The SELDnet architecture
# SELDnet 结构
# 一般是只改这里，因为其他的也不会改，也改不了

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed

'''
该类实现了自定义的均方误差损失函数。
功能：
初始化函数 __init__ 中通过 nn.MSELoss(reduction='none') 创建了一个均方误差损失函数对象 self._each_loss。
_each_calc 方法计算了输入张量 output 和目标张量 target 之间的均方误差损失，并对第二维求平均。
输入和输出：
输入：output 和 target 分别代表模型的输出和目标值。它们都是张量（tensor）类型的数据。
输出：该类的 __call__ 方法返回一个标量值 loss，代表计算得到的均方误差损失值。
具体流程：
    首先从 target 张量中提取出若干子张量，例如 target_A0、target_B0 等。
    然后根据提取的子张量，进行一系列的张量拼接操作，得到新的张量，如 target_A0A0A0、target_B0B0B1 等。
    对模型输出 output 进行形状变换，以便与之后的计算相匹配。
    根据一定的规则组合 target 子张量，得到 pad4A、pad4B、pad4C 三个新的张量。
    分别计算了 13 个损失值 loss_0 到 loss_12，然后找出这些损失值中最小的索引位置。
    最后根据最小损失的索引位置，从这 13 个损失值中加权组合得到最终的损失值 loss。
'''
class MSELoss_ADPIT(object): # 辅助重复排列不变训练，但我没看懂,  # 定义一个类的初始化方法，用于创建对象时进行初始化操作
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        # 创建一个名为 _each_loss 的属性，用 nn.MSELoss 创建一个均方误差损失函数对象，reduction='none' 表示不对损失值进行汇总
        self._each_loss = nn.MSELoss(reduction='none')

    # 定义一个方法 _each_calc，用于计算输出和目标之间的均方误差损失并返回平均值
    def _each_calc(self, output, target):
        # 调用之前创建的 _each_loss 对象，计算输出和目标之间的均方误差，然后沿着第二维度对损失值取平均
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    # 于计算损失
    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar

        13（=1+6+6）种可能组合的辅助重复排列不变训练
        Args：
        输出：[batch_size，frames，num_track*num_axis*num_class=3*3*12]
        目标:[batch_size, frames，num_track_dummy=6，num_axis=4，num_class=12]
        返回:
            损失：标量
        """
        # 计算一系列的目标值
        # 计算第一个目标值，取target中第0行，第0列的元素与第0行，从第1列到最后一列的元素逐个相乘
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class,A0，没有来自同一类的ov， [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        # 计算第二个目标值，取target中第1行，第0列的元素与第1行，从第1列到最后一列的元素逐个相乘
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class，B0，ov，具有来自同一类的2个源
        # 计算第三个目标值，取target中第2行，第0列的元素与第2行，从第1列到最后一列的元素逐个相乘
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        # 计算第四个目标值，取target中第3行，第0列的元素与第3行，从第1列到最后一列的元素逐个相乘
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class，C0，ov，具有来自同一类的3个源
        # 计算第五个目标值，取target中第4行，第0列的元素与第4行，从第1列到最后一列的元素逐个相乘
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        # 计算第六个目标值，取target中第5行，第0列的元素与第5行，从第1列到最后一列的元素逐个相乘
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        # 拼接目标值,通过在不同的维度上进行拼接，生成了一系列新的tensor对象。这些新的tensor对象可以用于后续的计算和处理。
        # 将target_A0与自身拼接，得到一个新的tensor，维度为[batch_size, sequence_length, 3, feature_dim]
        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), A的1个排列（没有来自同一类的ov），[batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)，B的6个置换（ov与来自同一类的2个源）
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)，C的6个置换（ov与来自同一类的3个源）
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        # 使用reshape函数对output tensor进行形状改变，让其与target_A0A0A0 tensor的形状相匹配。通过指定新的维度参数，将output tensor重塑为与target_A0A0A0 tensor相同维度的形状。这样做可以确保两个 tensor 在后续的计算中能够正确对齐和操作。
        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, 输出设置为与目标形状相同，[batch_size, frames, num_track*num_axis=3*3, num_class=12]
        
        # 对一系列损失进行计算。首先，分别计算了三个新的tensor：pad4A、pad4B和pad4C，它们是由不同的原始tensor相加得到的。
        # 然后，使用self._each_calc函数计算了一系列损失值，每个损失都是将output tensor与特定的加和tensor相加后传入函数进行计算得到的。这些加和tensor包括了不同组合的target tensor和对应的pad tensor。计算出的损失值分别赋值给了loss_0到loss_12变量。
        # 通过这些计算，可以得到一组用于评估模型性能的损失值，以便进行模型训练和优化。
        # 计算损失
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        # 使用self._each_calc函数计算output和target_A0A0A0 + pad4A的损失loss_0
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target 填充target_B0B0B1和target_C0C1C2，以避免将零设置为目标
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2 填充target_A0A0A0和target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1 填充target_A0A0A0和target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        # 找到最小的损失值的索引
        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        # 计算最终损失，根据最小损失值的索引，通过加权平均的方式计算得到最终的损失值loss。对于每个损失值，只有与最小损失值对应的索引相等时才会参与加权平均计算，最终求得的是各损失值的加权平均值作为最终损失。
        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        # 返回计算的损失
        return loss

# __init__方法：初始化函数，在创建类实例时被调用，用于初始化网络的结构和参数。其中，通过nn.Conv2d创建了一个二维卷积层对象self.conv，并设置了输入通道数、输出通道数、卷积核大小、步长和填充方式；通过nn.BatchNorm2d创建了一个二维批归一化层对象self.bn，并设置了输出通道数。
# forward方法：前向传播函数，在调用类实例的时候被自动调用，定义了数据从输入到输出的流向。在该方法中，输入的数据x首先经过卷积层self.conv，然后经过批归一化层self.bn，最后经过激活函数ReLU进行非线性变换，处理后的数据作为函数的返回值。
class ConvBlock(nn.Module): # 卷积模块
    # 初始化函数，定义网络结构
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__() # 继承父类
        '''
        Conv2d表示二维输入的卷积:
            in_channels：网络输入的通道数。
            out_channels：网络输出的通道数。
            kernel_size：卷积核的大小，如果该参数是一个整数q，那么卷积核的大小是qXq。
            stride：步长。是卷积过程中移动的步长。默认情况下是1。一般卷积核在输入图像上的移动是自左至右，自上至下。如果参数是一个整数那么就默认在水平和垂直方向都是该整数。如果参数是stride=(2, 1),2代表着高（h）进行步长为2，1代表着宽（w）进行步长为1。
            padding：填充，默认是0填充。
        '''
        # 创建卷积层对象，设置输入通道数、输出通道数、卷积核大小、步长和填充方式
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # 对输入的四维数组进行批量标准化处理
        # 创建批归一化层对象，设置输出通道数
        self.bn = nn.BatchNorm2d(out_channels)

    # 前向传播函数，定义数据流向
    def forward(self, x): # 前馈网络
        # 数据流向：卷积 -> 批归一化 -> 激活函数ReLU
        x = F.relu(self.bn(self.conv(x)))
        # 返回处理后的数据
        return x

# 该类是一个神经网络模块，用于生成位置编码。
# __init__方法：初始化函数，在创建类实例时被调用，用于初始化网络的结构和参数。在该方法中，首先创建一个形状为(max_len, d_model)的全零张量pe，然后依次计算位置信息、计算位置编码的奇偶索引位置上的sin值和cos值，并将最终得到的位置编码张量pe注册为模型的缓冲区。
# forward方法：前向传播函数，在调用类实例的时候被自动调用，定义了数据从输入到输出的流向。在该方法中，返回处理后的位置编码张量，只取前x.size(1)个位置的编码信息。
class PositionalEmbedding(nn.Module):  # Not used in the baseline，未在baseline中使用，位置嵌入/位置集成？
    # 初始化函数，用于创建对象时的初始化操作
    def __init__(self, d_model, max_len=512):
        super().__init__() # 继承父类

        # Compute the positional encodings once in log space.
        # 在日志空间中计算一次位置编码。
        # 创建一个形状为(max_len, d_model)的全零张量pe
        pe = torch.zeros(max_len, d_model).float()
        # 设置pe张量的梯度计算为False
        pe.require_grad = False

        # 创建一个表示位置信息的张量，shape为(max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算位置编码的除数项
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 计算位置编码中的奇数索引位置上的sin值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算位置编码中的偶数索引位置上的cos值
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展维度，将pe张量增加一个维度作为batch维度
        pe = pe.unsqueeze(0)
        # 将pe张量注册为模型的缓冲区
        self.register_buffer('pe', pe)

    # 前向传播函数，定义数据流向
    def forward(self, x): # 前馈网络
        # 返回处理后的位置编码张量，只取前x.size(1)个位置的编码信息
        return self.pe[:, :x.size(1)]

'''
定义了一个名为SeldModel的神经网络模型，用于语音事件定位和检测任务。
功能：
    初始化函数 (__init__)：
        初始化神经网络模型的各个组件，包括卷积块、GRU 网络、多头自注意力块和全连接层等。
        接受输入特征的形状 (in_feat_shape)、输出形状 (out_shape) 和其他参数 (params)。
        构建并配置网络的各个模块，如卷积块、GRU 网络、多头自注意力块等。
    前向传播函数 (forward)：
        对输入数据进行前向传播操作，经过卷积、GRU 网络、多头自注意力和全连接层的处理，最终生成预测结果。
    输入：
        x：输入数据，通常是一个张量，代表输入的特征数据。
    输出：
        doa：经过神经网络处理后得到的预测结果，通常代表语音事件的定位和检测结果。
    在整个流程中，输入数据会经过一系列的处理和转换，包括卷积操作、GRU 网络处理、多头自注意力机制以及全连接层的计算，最终得到预测结果 doa。
这个模型的目的是根据输入的特征数据，对语音事件进行定位和检测，以实现相关的任务。
'''
class SeldModel(torch.nn.Module): # SELD模型
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__() # 继承父类
        self.nb_classes = params['unique_classes'] # 获取数据集中声音事件的类别总数
        self.params=params # 获取数据的数据集参数、特征参数等等
        self.conv_block_list = nn.ModuleList() # # 创建一个空的模块列表，用于存储卷积块
        if len(params['f_pool_size']): # 如果池化尺寸列表不为空
            for conv_cnt in range(len(params['f_pool_size'])): # 遍历池化尺寸列表
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt'])) # 添加卷积块到模块列表
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))) # 添加最大池化层到模块列表
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))# 添加二维Dropout层到模块列表

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size']))) # 计算GRU网络的输入维度
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True) # 创建一个双向GRU网络

        # self.pos_embedder = PositionalEmbedding(self.params['rnn_size'])

        self.mhsa_block_list = nn.ModuleList() # 创建一个空的模块列表，用于存储多头自注意力块
        self.layer_norm_list = nn.ModuleList() # 创建一个空的模块列表，用于存储Layer Norm层
        for mhsa_cnt in range(params['nb_self_attn_layers']): # 遍历自注意力层数量
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=params['nb_heads'], dropout=params['dropout_rate'],  batch_first=True)) # 添加多头自注意力块到模块列表
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size'])) # 添加Layer Norm层到模块列表

        self.fnn_list = torch.nn.ModuleList() # 创建一个空的模块列表，用于存储全连接层
        if params['nb_fnn_layers']: # 如果全连接层数量不为0
            for fc_cnt in range(params['nb_fnn_layers']): # 遍历全连接层数量
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True)) # 添加全连接层到模块列表
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True)) # 添加输出层到模块列表
    
    '''
    遍历卷积块列表，对输入数据进行卷积处理。
    对张量进行维度转置和形状改变操作。
    使用GRU网络处理输入序列，并对输出进行tanh激活，并进行元素级别的乘法操作。
    遍历多头自注意力块列表，处理输入张量并进行相加操作，最后使用Layer Norm层处理张量。
    遍历全连接层列表，对张量进行全连接处理。
    最后使用输出层处理张量，并进行tanh激活，返回结果。
    '''
    def forward(self, x): # 前馈网络
        """input输入: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)): # 遍历卷积块列表
            x = self.conv_block_list[conv_cnt](x) # 使用卷积块处理输入数据

        x = x.transpose(1, 2).contiguous() # 转置张量的维度
        x = x.view(x.shape[0], x.shape[1], -1).contiguous() # 改变张量的形状
        (x, _) = self.gru(x) # 使用GRU网络处理输入序列
        x = torch.tanh(x) # 对输出进行tanh激活
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2] # 拆分输出张量并进行元素级别的乘法

        # pos_embedding = self.pos_embedder(x)
        # x = x + pos_embedding
        
        for mhsa_cnt in range(len(self.mhsa_block_list)): # 遍历多头自注意力块列表
            x_attn_in = x  # 复制输入张量
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in) # 使用多头自注意力块处理输入张量
            x = x + x_attn_in # 将原始输入张量与处理后的张量相加
            x = self.layer_norm_list[mhsa_cnt](x) # 使用Layer Norm层处理张量

        for fnn_cnt in range(len(self.fnn_list) - 1): # 遍历全连接层列表
            x = self.fnn_list[fnn_cnt](x) # 使用全连接层处理张量
        doa = torch.tanh(self.fnn_list[-1](x)) # 使用输出层处理张量并进行tanh激活
        return doa # 返回结果

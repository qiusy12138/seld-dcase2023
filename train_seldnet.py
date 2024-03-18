#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
# 一个训练SELDnet的包装器脚本。当早期停止度量-SELD错误停止改进时，训练停止。
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import seldnet_model
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model 

'''
函数接受两个参数：accdoa_in（包含音频事件定位与检测信息的数组）和nb_classes（类别数量）。
通过对accdoa_in进行切片操作，将其分为x、y、z坐标信息。
使用平方运算符**计算每个位置点的模长，即欧氏距离的平方。
对于每个位置点的模长，比较是否大于0.5，得到一个布尔数组，表示音频事件的发生与否。
最后返回音频事件的二值标签（sed）和原始的accdoa_in标签。
'''
def get_accdoa_labels(accdoa_in, nb_classes):
    # 从accdoa_in中提取x、y、z坐标信息，分别为前nb_classes列、接下来的nb_classes列、剩余的列
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    # 计算每个位置点的模长，即欧氏距离
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

    # 返回音频事件的二值标签（sed）和原始的accdoa_in标签
    return sed, accdoa_in


'''
这函数功能是根据输入的音频事件定位与检测信息（accdoa_in）和类别数量（nb_classes），提取每个音频事件的二值标签（sed）和方位角标签（doa）。
输入：
    accdoa_in：包含音频事件定位与检测信息的数组。
    nb_classes：类别数量。
输出：
    sed0：第一个音频事件的二值标签。
    doa0：第一个音频事件的方位角标签。
    sed1：第二个音频事件的二值标签。
    doa1：第二个音频事件的方位角标签。
    sed2：第三个音频事件的二值标签。
    doa2：第三个音频事件的方位角标签。
具体步骤如下：
    1. 对于第一个音频事件：
        从accdoa_in中提取x、y、z坐标信息，分别赋值给变量x0、y0、z0。
        计算每个位置点的模长，得到二值标签sed0，通过判断是否大于0.5来确定。
        提取方位角信息，赋值给变量doa0。
    2. 对于第二个音频事件：
        从accdoa_in中提取x、y、z坐标信息，分别赋值给变量x1、y1、z1。
        计算每个位置点的模长，得到二值标签sed1。
        提取方位角信息，赋值给变量doa1。
    3. 对于第三个音频事件：
        从accdoa_in中提取x、y、z坐标信息，分别赋值给变量x2、y2、z2。
        计算每个位置点的模长，得到二值标签sed2。
        提取方位角信息，赋值给变量doa2。
    4. 最后，将每个音频事件的二值标签（sed）和方位角标签（doa）作为结果返回。
'''
def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]

    位置参数：
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: 标量
    返回：
        sed的X轴坐标参数：[batch_size, frames, num_class=12]
        doa的X轴坐标参数：[batch_size, frames, num_axis*num_class=3*12]
    """
    
    # 提取第一个音频事件的x、y、z坐标信息
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    # 计算第一个音频事件每个位置点的模长，得到标签sed0
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    # 提取第一个音频事件的方位角信息
    doa0 = accdoa_in[:, :, :3*nb_classes]

    # 提取第二个音频事件的x、y、z坐标信息
    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    # 计算第二个音频事件每个位置点的模长，得到标签sed1
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    # 提取第二个音频事件的方位角信息
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    # 提取第三个音频事件的x、y、z坐标信息
    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    # 计算第三个音频事件每个位置点的模长，得到标签sed2
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    # 提取第三个音频事件的方位角信息
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    # 返回每个音频事件的二值标签（sed）和方位角标签（doa）
    return sed0, doa0, sed1, doa1, sed2, doa2


'''
这函数功能是判断两个音频事件的位置是否相似。
输入：
    sed_pred0：第一个音频事件的二值标签。
    sed_pred1：第二个音频事件的二值标签。
    doa_pred0：第一个音频事件的方位角标签。
    doa_pred1：第二个音频事件的方位角标签。
    class_cnt：类别计数。
    thresh_unify：统一阈值。
    nb_classes：类别数量。
输出：
    如果两个音频事件的位置相似，返回1；否则返回0。
  首先检查两个音频事件的二值标签是否都为1。
  如果两个音频事件的二值标签都为1，则进入条件判断。
  在满足条件的情况下，调用distance_between_cartesian_coordinates函数计算两个音频事件之间的空间距离，并将结果与给定的统一阈值thresh_unify进行比较。
  如果计算得到的距离小于阈值，则判断为相似位置，返回1；否则，判断为不同位置，返回0。
  如果两个音频事件的二值标签不都为1，直接返回0，表示不同位置。
'''
def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    # 检查两个音频事件的二值标签是否都为1
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            # 如果距离小于阈值，则判断为相似位置
            return 1
        else:
            # 如果距离大于等于阈值，则判断为不同位置
            return 0
    else:
         # 若两个音频事件的二值标签不都为1，则判断为不同位置
        return 0


'''
这函数用于测试音频事件检测和方位角回归模型在给定数据上的性能表现，并输出预测结果到文件中。
功能：
    1. 从给定的数据生成器中获取数据并进行测试。
    2. 使用模型进行前向传播，计算损失。
    3. 根据设定的参数判断是否需要进行多任务评估。
    4. 根据预测结果调整音频事件检测（SED）和方位角回归（DOA）标签。
    5. 生成输出文件，并记录每帧的类别和方位角预测。
输入：
    data_generator: 数据生成器，用于生成测试数据。
    model: 被测试的模型。
    criterion: 损失函数。
    dcase_output_folder: 输出文件夹路径。
    params: 包含各种参数的字典。
    device: 设备信息，指定在哪个设备上进行计算。
输出：
    无返回值，但会将测试结果输出到指定的文件夹中。
主要的流程是：
    1. 循环遍历数据生成器生成的数据和目标。
    2. 将数据和目标转换为张量并移动到指定设备上。
    3. 判断是否需要多任务评估，根据不同情况获取和调整音频事件检测和方位角回归标签。
    4. 根据预测结果构建输出文件路径，并记录每帧的类别和方位角预测到输出字典中。
'''
def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # 具有100ms跳长的60秒音频的帧数=600帧
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    # 一个批次中的帧数（batch_size*sequence_length）由以上所有600帧组成，其余帧为零填充
    test_filelist = data_generator.get_filelist() # 获取测试文件列表

    nb_test_batches, test_loss = 0, 0. # 初始化测试批次数和测试损失
    model.eval() # 设置模型为评估模式
    file_cnt = 0 # 初始化文件计数器
    with torch.no_grad(): # 关闭梯度计算
        for data, target in data_generator.generate(): # 遍历数据生成器产生的数据和目标
            # load one batch of data
            # 加载一批量的数据
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float() # 将数据和目标转换为张量，并移动到指定设备上

            # process the batch of data based on chosen model
            # 基于所选模型处理一批数据
            output = model(data) # 使用模型进行前向传播得到输出
            loss = criterion(output, target) # 计算损失
            if params['multi_accdoa'] is True: # 如果需要多任务评估
                # 获取多任务的音频事件检测（SED）和方位角回归（DOA）标签
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                # 将SED和DOA标签重新调整为2D形状
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                # 获取音频事件检测（SED）和方位角回归（DOA）标签
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                # 将SED和DOA标签重新调整为2D形状
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            # 将SELD结果转储到相应的文件
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv')) # 构建输出文件路径
            file_cnt += 1 # 文件计数器加一
            output_dict = {} # 初始化输出字典
            if params['multi_accdoa'] is True: # 如果需要多任务评估
                for frame_cnt in range(sed_pred0.shape[0]): # 遍历每个帧数
                    for class_cnt in range(sed_pred0.shape[1]): # 遍历每个类别
                        # determine whether track0 is similar to track1
                        # 确定track0是否与track1相似
                        # 使用三个不同的预测结果来确定相似的位置
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        # 根据flag来确定是否统一？unify怎么翻译才准确
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0: # 如果三个预测结果都不相似
                            if sed_pred0[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = [] # 初始化帧数对应的输出列表
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                            if sed_pred1[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = [] # 初始化帧数对应的输出列表
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                            if sed_pred2[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = [] # 初始化帧数对应的输出列表
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1: # 如果有一个预测结果相似
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = [] # 初始化帧数对应的输出列表
                            if flag_0sim1: # 如果第一个和第二个预测结果相似
                                if sed_pred2[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2 # 计算两个DOA预测的平均值
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_1sim2: # 如果第二个和第三个预测结果相似
                                if sed_pred0[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2 # 计算两个DOA预测的平均值
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_2sim0: # 如果第三个和第一个预测结果相似
                                if sed_pred1[frame_cnt][class_cnt]>0.5: # 如果SED预测大于0.5
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]]) # 将类别和DOA预测添加到输出列表中
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2 # 计算两个DOA预测的平均值
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2: # 如果有两个或以上的预测结果相似
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = [] # 初始化帧数对应的输出列表
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3 # 计算三个DOA预测的平均值
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]]) 
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break


        test_loss /= nb_test_batches

    return test_loss


'''
这函数实现了模型的训练过程，包括了前向传播、损失计算、反向传播更新参数等步骤，最终返回训练过程中的平均损失值。
功能：
    1. 循环遍历数据生成器产生的数据和目标。
    2. 将数据和目标转换为张量并移动到指定设备上。
    3. 对模型进行训练，计算损失并更新参数。
    4. 如果设置了quick_test参数，并且已经完成了指定数量的批次训练，则提前结束训练。
    5. 最终返回训练过程中的平均损失值。
输入：
    data_generator: 数据生成器，用于生成训练数据。
    optimizer: 优化器，用于更新模型参数。
    model: 要训练的模型。
    criterion: 损失函数，用于计算损失。
    params: 包含各种参数的字典。
    device: 设备信息，指定在哪个设备上进行计算。
输出：
    返回训练过程中的平均损失值。
流程：
    1. 将模型设置为训练模式。
    2. 遍历数据生成器产生的每个批次数据和对应的目标。
    3. 将数据和目标转换为张量并移动到指定设备上。
    4. 使用优化器将模型的梯度清零。
    5. 通过模型前向传播得到输出，并计算损失。
    6. 反向传播更新模型参数。
    7. 累加损失值并记录训练批次数。
    8. 如果设置了quick_test参数并达到指定批次数目，则提前结束训练。
    9. 计算平均损失值并返回。
'''
def train_epoch(data_generator, optimizer, model, criterion, params, device):
    # 初始化训练批次数和训练损失
    nb_train_batches, train_loss = 0, 0.
    # 将模型设置为训练模式
    model.train()
    # 遍历数据生成器产生的每个批次数据和对应的目标
    for data, target in data_generator.generate():
        # load one batch of data
        # 加载一批量的数据
        # 将数据和目标转换为张量并移动到指定设备上
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        # 使用优化器将模型的梯度清零
        optimizer.zero_grad()

        # process the batch of data based on chosen model
        # 依照所选的模型来处理一批量的数据
        # 通过模型前向传播得到输出
        output = model(data)

        # 计算损失
        loss = criterion(output, target)
        # 反向传播更新模型参数
        loss.backward()
        optimizer.step()

        # 累加损失值并记录训练批次数
        train_loss += loss.item()
        nb_train_batches += 1
         # 如果设置了`quick_test`参数并达到指定批次数目，则提前结束训练
        if params['quick_test'] and nb_train_batches == 4:
            break

    # 计算平均损失值
    train_loss /= nb_train_batches

    # 返回训练过程中的平均损失值
    return train_loss



'''
定义一个函数main，接收一个参数argv

主程序实现了一个深度学习模型的训练与评估流程，针对声音事件检测和方向性角度估计任务进行了模型的训练和评估。同时根据不同的数据集拆分方式进行了多次训练和评估，最终输出各项性能指标和结果。
1. 功能：
    检查命令行参数数量是否为3，如果不是，则打印出用法提示信息。
    根据命令行参数和设备情况确定是否使用GPU。
    根据指定的任务ID和作业ID加载参数。
    根据数据集路径选择测试、验证和训练数据集拆分方式。
    针对每个测试拆分，执行模型训练和评估，输出结果。
2. 输入：
    两个可选的命令行参数：<task-id> <job-id>
3. 输出：
    训练过程中的日志信息，包括模型配置、损失值、性能指标等。
    最佳模型的权重将被保存。
    测试集上的评估结果，包括损失值、SED（声音事件检测）指标、DOA（方向性角度估计）指标等。
    如果选择了macro平均方式，还会输出类别级别的评估结果。
'''
def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    用于训练声音事件定位和检测网络的主包装器。
    ：param argv参数：需要两个可选输入。
        第一个输入：task_id-（可选）在parameters.py中选择系统配置。
                    （默认）1-使用默认参数
        第二个输入：jobid-（可选）所有输出文件都将用这个唯一表示。
                    （默认）1
    """

    # argv是一个指向字符串数组的指针，每个字符串表示一个命令行参数。
    print(argv) # 打印传入的参数
    if len(argv) != 3: # 如果参数长度不等于3
        # 打印提示信息
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs') # 代码需要两个可选输入
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py') # <task-id>用于从parameter.py中选择用户定义的参数集
        print('Using default inputs for now') # 目前使用默认输入
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.') # <job-id>是用于输出文件名（模型、训练图）的唯一标识符。您可以为此使用任何数字或字符串。
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # 检查是否有CUDA可用，并设置设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    # 使用用户定义的参数集
    # 根据参数获取任务ID和参数
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # 获取作业ID
    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    # 训练设置
    # 初始化训练、验证和测试数据集拆分
    train_splits, val_splits, test_splits = None, None, None
    # 根据数据集年份设置数据集拆分
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]]
            
        else:
            print('ERROR: Unknown dataset splits') # 出错：未知数据集拆分
            exit()
    
   # 遍历测试数据集拆分并打印信息
    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        # 运行的唯一名称
        # 设置本地特征和输出类型
        loc_feat = params['dataset'] # 将params字典中的'dataset'键对应的值赋给loc_feat变量
        if params['dataset'] == 'mic': # 如果params字典中'dataset'键对应的值为'mic'
            if params['use_salsalite']: # 如果params字典中'use_salsalite'键对应的值为True
                loc_feat = '{}_salsa'.format(params['dataset']) # loc_feat设为'mic_salsa'
            else: # 否则loc_feat设为'mic_gcc'
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa' # 根据'multi_accdoa'键的值设置loc_output变量

        # 创建文件夹
        cls_feature_class.create_folder(params['model_dir']) # 调用create_folder方法在指定路径下创建文件夹
        # 生成唯一名称
        # 根据给定参数格式化生成唯一名称
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        # 构建模型名称
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name)) # 构建模型名称
        # 打印唯一名称
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        # 加载训练和验证数据，打印加载训练数据集提示信息
        print('Loading training dataset:') #加载训练数据集
        # 使用DataGenerator加载训练数据集
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        # 打印加载验证数据集提示信息
        print('Loading validation dataset:') # 加载验证数据集
        # 使用DataGenerator加载验证数据集
        # 创建一个验证数据生成器对象，使用指定的参数、切分数据和设置不打乱数据、按文件处理选项
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # Collect i/o data size and load model configuration
        # 收集输入/输出数据大小和加载模型配置
        # 调用训练数据生成器对象的get_data_sizes()方法，获取训练数据的输入和输出大小
        data_in, data_out = data_gen_train.get_data_sizes()
        
        # 创建SELD模型
        # 创建一个SELD模型对象，使用数据输入和输出大小以及参数初始化，并将其移动到指定设备上执行
        model = seldnet_model.SeldModel(data_in, data_out, params).to(device)
        # 在微调模式下加载模型权重
        # 如果处于微调模式，则加载预训练模型的权重到模型中
        if params['finetune_mode']:
            print('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights'])) # 在微调模式下运行。将模型初始化为权重-{}
            model.load_state_dict(torch.load(params['pretrained_model_weights'], map_location='cpu'))

        # 打印模型信息
        # 打印模型相关信息，包括输入输出大小和模型配置参数
        print('---------------- SELD-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out)) # 特征
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
            params['fnn_size'])) # 模型，dropout脱落率/丢弃率，池化大小，注意力模块，参数
        print(model)

        # Dump results in DCASE output format for calculating final scores
        # 以DCASE输出格式转储结果以计算最终分数
        # 创建输出文件夹
        # 创建一个输出文件夹用于保存验证结果，并打印输出路径
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder)) # 转储清晰记录值进入：{}

        # Initialize evaluation metric class
        # 初始化评估度量类
        # 计算SELD结果
        score_obj = ComputeSELDResults(params)

        # start training
        # 开始训练
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999 
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        # 初始化计算SELD结果的对象、最佳验证时期、错误率、F值等变量，以及优化器
        # 根据参数中的quick_test值确定nb_epoch的取值，然后使用Adam优化器来优化模型的参数
        # 根据多分类或单分类设置损失函数
        if params['multi_accdoa'] is True: # 如果参数中的'multi_accdoa'为True，则使用自定义的损失函数MSELoss_ADPIT
            criterion = seldnet_model.MSELoss_ADPIT()
        else: # 否则使用PyTorch中的均方误差损失函数nn.MSELoss
            criterion = nn.MSELoss()

        # 循环训练模型
        for epoch_cnt in range(nb_epoch): # 遍历训练周期数(nb_epoch)
            # ---------------------------------------------------------------------
            # TRAINING
            # 训练
            # ---------------------------------------------------------------------
            start_time = time.time() # 记录当前时间作为起始时间
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device) # 训练模型并计算训练损失
            train_time = time.time() - start_time # 计算训练时间

            # ---------------------------------------------------------------------
            # VALIDATION
            # 验证
            # ---------------------------------------------------------------------
            start_time = time.time() # 记录当前时间作为起始时间
            val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device) # 在验证集上测试模型并计算验证损失

            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            # 计算DCASE 2021指标-位置感知检测和类别感知定位得分
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder) # 获取验证结果指标

            val_time = time.time() - start_time # 计算验证时间
            
            # Save model if loss is good
            # 如果损失良好，则保存模型
            if val_seld_scr <= best_seld_scr: # 如果当前的SELD分数小于等于历史最佳的SELD分数
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr # 更新最佳验证时期和最佳指标
                torch.save(model.state_dict(), model_name) # 保存模型参数

            # Print stats
            # 打印训练和验证信息
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F/LE/LR/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                    best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR, best_seld_scr))
            )

            patience_cnt += 1 # 增加耐心计数器
            if patience_cnt > params['patience']: # 如果耐心计数器超过设定的耐心值
                break # 跳出训练循环

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # 对看不见的测试数据进行评估
        # ---------------------------------------------------------------------
        # 打印信息：加载最佳模型权重
        print('Load best model weights') # 加载最佳模型权重
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        # 打印信息：加载未见过的测试数据集
        print('Loading unseen test dataset:') # 加载看不见的测试数据集
        # 创建数据生成器对象data_gen_test，用于生成测试数据
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        # 以DCASE输出格式转储结果以计算最终分数
        # 构建文件夹路径dcase_output_test_folder
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        # 删除并重新创建文件夹dcase_output_test_folder
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        # 打印信息：将每个录音的测试结果保存到指定文件夹dcase_output_test_folder中
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder)) # 转储清晰记录值进入：{}

        # 调用test_epoch函数进行测试，并得到测试损失值test_loss
        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

        # 设置变量use_jackknife为True
        use_jackknife=True
        # 通过调用score_obj对象的get_SELD_Results方法，获取测试结果的各项评分指标
        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
        # 打印信息：测试损失值
        print('\nTest Loss') # 测试损失
        # 打印信息：SELD得分（用于早停的指标）
        print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        # 打印信息：SED指标：错误率和F-score
        print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        # 打印信息：DOA指标：定位误差和定位召回率
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''))
        # 判断average是否等于'macro'
        if params['average']=='macro':
            # 打印信息：在未见过的测试数据上的类别结果
            print('Classwise results on unseen test data') # 看不见的测试数据的分类结果
            print('Class\tER\tF\tLE\tLR\tSELD_score') 类别？的SELD得分
            # 遍历每个类别，打印各项评分指标
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                     cls_cnt,
                     classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))



if __name__ == "__main__":
    try:
        # 尝试执行主函数，并将命令行参数传入
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        # 如果出现ValueError或IOError异常，则捕获并退出程序，返回错误信息
        sys.exit(e)


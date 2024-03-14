# Extracts the features, labels, and normalizes the development and evaluation split features.
# 提取特征、标签并规范化开发和评估拆分特征。

# 加载项目中的其他文件：特征类别cls_feature_class和参数parameters
import cls_feature_class
import parameters
# 加载Python中自带的工具包sys
import sys


def main(argv):            # argv保存命令行输入的参数值,argv[0]指向程序自身运行目录路径和程序名，,argv[1]指向程序在DOS命令中执行程序名后的第一个字符串,argv[2]指向第二个字符串
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # 需要一个输入：task-id（任务-id）来对应于parameter.py文件中给定的配置。
    # Extracts features and labels relevant for the task-id
    # 提取与task-id（任务-id）相关的特征和标签
    # It is enough to compute the feature and labels once. 
    # 计算一次特征和标签就足够了。

    # use parameter set defined by user
    # 使用用户定义的参数集，这里的参数集是给定的数据集
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id) # get_params函数返回的是参数字典，参数组主要包含以下几类参数：数据集参数、特征参数、模型类别、DNN模型参数、指标

    # -------------- Extract features and labels for development set -----------------------------
    # -------------- 提取开发集（给定的数据集）的特征和标签 -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    # # Extract features and normalize them
    # # 提取特征并对其进行规范化
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels
    # # 提取标签
    dev_feat_cls.extract_all_labels()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


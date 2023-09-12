# chatGPT 分析报告
## 翻译 private_upload/2023-09-12-08-19-18/README.md.part-0.md

# MONAI 和 nnU-Net 集成

[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 是一个专门为医学图像分割设计的开源深度学习框架。nnU-Net 是一个最先进的深度学习框架，专为医学图像分割而定制。它基于流行的 U-Net 架构，并结合了各种先进的功能和改进，如级联网络、新颖的损失函数和预处理步骤。nnU-Net 还提供了一个易于使用的界面，允许用户快速训练和评估其分割模型。nnU-Net 已被广泛应用于各种医学图像应用，包括脑分割、肝脏分割和前列腺分割等。该框架在各种基准数据集和挑战中一直保持着最先进的性能，证明了它在推动医学图像分析方面的效果和潜力。

nnU-Net 和 MONAI 是两个强大的开源框架，提供先进的医学图像分析工具和算法。这两个框架在研究界得到了广泛的认可，并且许多研究人员已经在使用这些框架开发新的创新的医学图像应用。

nnU-Net 是一个为医学图像分割任务提供标准化流程的框架。而 MONAI 则是一个提供了全面工具集的医学图像分析框架，包括预处理、数据增强和深度学习模型等。它还基于 PyTorch，并提供了广泛的预训练模型，以及用于模型训练和评估的工具。nnUNet 和 MONAI 的集成可以为医学影像领域的研究人员带来多方面的好处。通过结合 nnUNet 提供的标准化流程和 MONAI 提供的全面工具集，研究人员可以充分发挥两个框架的优势。

总的来说，nnU-Net 和 MONAI 的集成可以为医学影像领域的研究人员提供重要的好处。通过结合两个框架的优势，研究人员可以加快研究进展，开发出复杂医学影像挑战的新的创新解决方案。

## nnU-Net V2 的新特性

nnU-Net 最近发布了一个新版本，即 nnU-Net V2。以下是一些变化：
- 重新设计的代码库：nnU-Net v2 在代码库结构方面进行了重大改变，使其更易于浏览和理解。代码库已经模块化，并且文档已经改进，可以更容易地与其他工具和框架集成。
- 新特性：nnU-Net v2 引入了一些新的[特性](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/changelog.md)，包括：
  - 基于区域的公式和 sigmoid 激活函数；
  - 跨平台支持；
  - 多 GPU 训练支持。

总的来说，nnU-Net v2 引入了重大改进和新特性，使其成为一个功能强大、灵活的医学图像分割深度学习框架。凭借其易于使用的界面、模块化的代码库和先进的特性，nnU-Net v2 正在推动医学图像分析领域的发展，并改善患者的结果。

## 集成是如何工作的？
作为集成的一部分，我们引入了一个名为 `nnUNetV2Runner` 的新类，该类利用了官方 nnU-Net 代码库中提供的 Python API。`nnUNetV2Runner` 提供了几个对 MONAI 的普通用户有用的核心功能。
- 新的类在高级别上提供了 Python API，用于便利大多数 nnU-Net 组件，如模型训练、验证、合成等；
- 用户只需要提供最小限度的输入，即大多数用于 3D 医学图像分割的 MONAI 教程指定的输入。新的类会自动处理数据转换，以准备满足 nnU-Net 要求的数据，这大大节省了用户准备数据集的时间；
- 此外，我们还启用了更多 GPU 资源的用户可以自动并行分配模型训练任务。由于 nnU-Net 默认要求训练 20 个分割模型，将模型训练分布到更大的资源上可以显著提高整体效率。例如，具有 8 个 GPU 的用户可以使用新的类自动将模型训练速度提高 6 到 8 倍。

## 在公共数据集上的基准结果

在本部分，我们展示了我们的 `nnUNetV2Runner` 的结果以及官方 nnU-Net 代码库在各种公共数据集上的结果。目标是验证我们的 `nnUNetV2Runner` 实现是否达到了原生 nnU-Net 的性能。

### 数据集

1. [BraTS21](http://braintumorsegmentation.org/)：RSNA-ASNR-MICCAI BraTS 2021 挑战利用多机构术前基线多参数磁共振成像 (mpMRI) 扫描，并专注于评估（任务 1）用于分割 mpMRI 扫描中内在异质性脑胶质母细胞瘤的最先进方法的性能。任务 1 的目标是评估不同分割方法在 500 个案例上的性能，这些案例具有 15 个器官的注释。任务 2 在任务 1 的基础上扩展，除了 CT 扫描外，还包括 MRI 扫描。在这种“跨模态”设置下，单个算法必须从 CT 和 MRI 扫描中分割腹部器官。该任务还提供了额外的 100 个具有相同类型注释的 MRI 扫描。

下表显示了各数据集上基于完整分辨率的 3D U-Net 的结果，其中包括原生 nnU-Net 和 `nnUNetV2Runner` 的性能。

| 任务 | 原生 nnU-Net | `nnUNetV2Runner` |
|-----------------|-----------------|-----------------|
| BraTS21 | 0.92 | 0.94 |
| AMOS22 (任务 1) | 0.90 | 0.90 |
| AMOS22 (任务 2) | 0.89 | 0.89 |

## 步骤

### 安装

安装说明详见 [这里](docs/install.md)。

### 数据集和数据列表准备

用户需要为新任务提供一个数据列表（".json" 文件）和数据根目录。通常，有效的数据列表需要遵循 [Medical Segmentation Decathlon (MSD)](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) 中的格式。

在[这个教程](../auto3dseg/notebooks/msd_datalist_generator.ipynb)中，我们提供了下载 [MSD Spleen 数据集](http://medicaldecathlon.com) 并准备数据列表的示例步骤。
下面我们假设数据集已下载到 `/workspace/data/Task09_Spleen`，数据列表在当前目录下。

### 用 `nnUNetV2Runner` 使用最小输入运行

## 翻译 private_upload/2023-09-12-08-19-18/README.md.part-1.md

创建数据列表后，用户可以创建一个简单的“input.yaml”文件（如下所示），作为**nnUNetV2Runner**的最小输入。

```
modality: CT
datalist: "./msd_task09_spleen_folds.json"
dataroot: "/workspace/data/Task09_Spleen"
```

注意：对于多模态输入，请查看[常见问题解答](#FAQ)部分。

用户还可以将目录变量的值设置为“input.yaml”中的选项，如果需要指定任何目录。

```
dataset_name_or_id: 1 ＃任务特定的整数索引（可选）
nnunet_preprocessed: "./work_dir/nnUNet_preprocessed" ＃用于存储预处理数据的目录（可选）
nnunet_raw: "./work_dir/nnUNet_raw_data_base" ＃用于存储格式化原始数据的目录（可选）
nnunet_results: "./work_dir/nnUNet_trained_models" ＃用于存储训练模型检查点的目录（可选）
```

一旦提供了最小输入信息，用户可以使用以下命令自动启动整个nnU-Net流程的过程（从模型训练到模型集成）。

```bash
python -m monai.apps.nnunet nnUNetV2Runner run --input_config='./input.yaml'
```

为了进行实验和调试，用户可能希望在nnU-Net流程中设置训练的轮数。
我们的集成提供了一个可选参数`trainer_class_name`来指定训练轮数，如下所示：

```bash
python -m monai.apps.nnunet nnUNetV2Runner run --input_config='./input.yaml' --trainer_class_name nnUNetTrainer_1epoch
```

支持的`trainer_class_name`为：
- nnUNetTrainer（默认）
- nnUNetTrainer_1epoch
- nnUNetTrainer_5epoch
- nnUNetTrainer_10epoch
- nnUNetTrainer_20epoch
- nnUNetTrainer_50epoch
- nnUNetTrainer_100epoch
- nnUNetTrainer_250epoch
- nnUNetTrainer_2000epoch
- nnUNetTrainer_4000epoch
- nnUNetTrainer_8000epoch

### 使用```nnUNetV2Runner```运行nnU-Net模块

```nnUNetV2Runner```提供了一体化API来执行流程，以及访问nnU-Net V2底层组件的API。下面是不同组件的命令。

```bash
## [component] 转换数据集
python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config "./input.yaml"

## [component] 实验计划和数据预处理
python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

## [component] 使用所有可用GPU训练所有20个模型
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml"

## [component] 使用所有可用GPU训练单个模型
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0

## [component] 分布式训练20个模型，利用指定的GPU设备0和1
python -m monai.apps.nnunet nnUNetV2Runner train --input_config "./input.yaml" --gpu_id_for_all 0,1

## [component] 找到最佳配置
python -m monai.apps.nnunet nnUNetV2Runner find_best_configuration --input_config "./input.yaml"

## [component] 预测、集成和后处理
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml"

## [component] 仅预测
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
    --run_ensemble false --run_postprocessing false

## [component] 仅集成
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
    --run_predict false --run_postprocessing false

## [component] 仅后处理
python -m monai.apps.nnunet nnUNetV2Runner predict_ensemble_postprocessing --input_config "./input.yaml" \
    --run_predict false --run_ensemble false
```

为了在多GPU训练中利用PyTorch DDP，我们提供了以下命令，以便在特定折叠上训练特定模型：

```bash
## [component] 单个模型的多GPU训练
python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" \
    --config "3d_fullres" \
    --fold 0 \
    --gpu_id 0,1
```

我们提供了一个替代API，用于从[MSD挑战赛](http://medicaldecathlon.com/)构建数据集，以满足nnU-Net的要求，参考[链接](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#how-to-use-decathlon-datasets)。

```bash
## [component] 转换msd数据集
python -m monai.apps.nnunet nnUNetV2Runner convert_msd_dataset --input_config "./input.yaml" --data_dir "/workspace/data/Task09_Spleen"
```

## 常见问题解答

常见问题和答案可以在此处找到[here](docs/faq.md)。


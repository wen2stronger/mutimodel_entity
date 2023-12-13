import os
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

def main(config):
    # 从bulildloader 生成数据集
    # 根据配置对象构建训练和验证数据集，以及相应的数据加载器
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # 使用日志记录器 logger 打印创建模型的信息，包括模型类型和名称，这些信息从 config 对象中获取
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # 调用 build_model 函数来根据配置创建模型
    model = build_model(config)
    # 打印模型的详细信息
    logger.info(str(model))

    # 计算模型的可训练参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # 检查模型是否有一个方法或属性名为 flops
    if hasattr(model, 'flops'):
        flops = model.flops()  # 如果有 flops 属性，调用它来计算模型的浮点运算数
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()  # 将模型移动到GPU上
    model_without_ddp = model  # 创建一个指向原始模型的引用

    optimizer = build_optimizer(config, model)  # 创建优化器
    # 将模型封装为分布式数据并行模型，用于在多个GPU上训练
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    # 创建一个梯度缩放器，这通常用于混合精度训练
    loss_scaler = NativeScalerWithGradNormCount()

    # 创建学习率调度器
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 损失函数配置
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    # 代码检查是否需要从之前的检查点自动恢复训练
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # 从指定的检查点恢复
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    # 加载预训练模型
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    # 吞吐量模式：代码将计算并记录数据处理的吞吐量
    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    # 开始训练
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)  # 设置训练数据加载器的采样器以便于进行数据洗牌

        # 这行代码执行一次训练迭代
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        # 这行代码检查是否需要保存模型的检查点
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        # 在每个epoch结束时，使用验证集评估模型的性能
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # 更新记录的最高准确率
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    # print(args)
    # print('-'*90)
    # print(config)
    return args, config



def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()  # 这行设置模型为训练模式
    optimizer.zero_grad()  # 在开始新的梯度计算之前，将模型参数的梯度清零

    num_steps = len(data_loader)  # 计算数据加载器中批次（batch）的总数
    batch_time = AverageMeter()  # 初始化一个用于跟踪批处理时间的计数器
    loss_meter = AverageMeter()  # 初始化一个用于跟踪平均损失的计数器
    norm_meter = AverageMeter()  # 初始化一个用于跟踪梯度范数的计数器
    scaler_meter = AverageMeter()  # 初始化一个用于跟踪损失缩放比例的计数器

    start = time.time()
    end = time.time()

    # 深度学习中典型的复杂训练循环的一个例子，包含了梯度累积、混合精度训练和动态学习率调整等实践
    for idx, (samples, targets) in enumerate(data_loader):  # 遍历数据加载器中的每个批次，samples 是输入数据，targets 是对应的目标值
        samples = samples.cuda(non_blocking=True)  # 将输入数据移动到GPU上进行计算
        targets = targets.cuda(non_blocking=True)  # 将目标数据也移动到GPU上

        # 如果提供了混合函数（一种数据增强技术），则对数据进行混合
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 启用自动混合精度（AMP），如果配置允许，可以提高训练效率和减少内存使用
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)  # 将输入数据通过模型获得输出结果
        loss = criterion(outputs, targets)  # : 这行代码通过将模型的输出与实际目标（真实值）进行比较来计算损失
        loss = loss / config.TRAIN.ACCUMULATION_STEPS  # 损失值被分配到多个累积步骤上

        # this attribute is added by timm on one optimizer (adahessian)
        # 有关is_second_order的代码块检查是否使用的优化器具有is_second_order属性
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # 这行代码缩放梯度，以防止数值不稳定或梯度消失/爆炸的问题
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        # 这个条件检查是否到了更新梯度的时候
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()  #  重置模型参数的梯度
            # 根据完成的步数调整学习率
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        # 获取混合精度训练中损失缩放器的当前比例
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()  # 等待所有CUDA内核完成

        loss_meter.update(loss.item(), targets.size(0))  # 使用当前的损失和批处理大小更新一个度量工具
        if grad_norm is not None:  # loss_scaler return None if not update 检查是否计算了梯度范数并更新相应的度量工具
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)  # 使用损失比例值更新另一个度量工具
        batch_time.update(time.time() - end)  # 使用处理批处理所需的时间更新计时度量工具
        end = time.time()

        # 这段代码是一个训练循环中的一部分，用于在特定的迭代频率下输出训练过程的信息。
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# 用于评估（验证）机器学习模型的性能
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        # print(output)
        # time.sleep(1)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc2)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    # 从parse 生成参数，配置
    args, config = parse_option()

    # 以下为分布式训练内容 先注视了
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    ##################################################################################
    # 本地单卡训练也需要初始化 process group
    torch.distributed.init_process_group('gloo', init_method='file://D:/AI/swin-transformer/tempfile', rank=0,
                                         world_size=1)
    # 设置阻塞，保证所有进程同步
    torch.distributed.barrier()
    # 随机种子，保证代码可复现性，seed能保证随机出来的数据一致
    seed = config.SEED
    # cpu 随机种子
    torch.manual_seed(seed)
    # gpu 随机种子
    torch.cuda.manual_seed(seed)
    #　ｎｐ数组随机种子
    np.random.seed(seed)
    # py 随机种子
    random.seed(seed)
    # 预处理模型 寻找最优卷积实现算法
    cudnn.benchmark = True
    ##########################################################################################################
    #取消了学习率的线性缩放
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR
    # gradient accumulation also need to scale the learning rate
    # 当梯度累加值大于1的时候，lr还根据累加步数缩放
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    #########################################################################################################
    # 创建output文件夹
    os.makedirs(config.OUTPUT, exist_ok=True)
    # 创建日志
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    # 如果当期是主进程， 在output中 保存配置
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(config)

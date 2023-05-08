import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from building_blocks.builder import ConvBNRelu, Identity, Upsample


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def gather_bn_weights(model):
    size_list = []
    bn_modules = []
    for idx, m in enumerate(model.module_list):
        if 2 <= idx <= 17 and isinstance(m, ConvBNRelu):
            bn_module = m[1]
            bn_modules.append(bn_module)
            size_list.append(bn_module.weight.data.shape[0])

    #size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx] # [32, 64, 128,...]
    # 将所有BN的所有channel展开放一起
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = bn_modules[idx].weight.data.abs().clone()
        index += size

    return bn_weights


def pack_filter_para(start_id, end_id, num_filters):
    para = [[] for i in range(end_id-start_id+1)]
    for i in num_filters:
        if start_id <= i[0] <= end_id:
            para[i[0]-start_id].append(i[1])
    for i in para:
        if len(i) == 0:
            i.append(None)
    return para


def generate_searchspace(para, first_input):
    channel_size = []
    input_shape = [first_input]
    
    for i in para:
        if i[0] is not None:
            channel_size.append(i[0])
    
    for i in range(len(channel_size)-1):
        input_shape.append(channel_size[i])
        
#     for i in para:
#         channel_size.append(i[-1])
#         if len(i) > 2:
#             prune.append((i[0], i[1]))
#         else:
#             prune.append(None)
    
#     if channel_size[0] is None:
#         channel_size[0] = first_input

#     for i in range(1, len(channel_size)):
#         prev = channel_size[i-1]
#         current = channel_size[i]
#         if current is None:
#             channel_size[i] = prev

#     for i, j in enumerate(channel_size):
#         if i != len(channel_size)-1:
#             input_shape.append(j)
    return input_shape, channel_size


class BNOptimizer:
    @staticmethod
    def updateBN(sr_flag, model, s):
        if sr_flag:
            for idx, module in enumerate(model.module_list):
                if 1 <= idx <= 17:
                    for m in module.modules():
                        if isinstance(m, ConvBNRelu):
                            bn_module = m[1]  # 1对应BN
                            bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def BN_preprocess(model):

    fusion_modules = [[], [], [], []]  # [[(false, pwl),(tru, pwl),...()], [head26], [head13]]
    fusion_modules_in_block = [[], [], [], []] # [[(1, pw, dw),(2, pw, dw),...], ]
    
    fusion_modules[0].append((None, model.module_list[0]))  # backbone第一层不参与剪枝，仅用于占位

    fusion_modules[1].append((None, model.module_list[12])) # fpn与backbone的2个连接处
    fusion_modules[1].append((None, model.module_list[14]))

    fusion_modules[2].append((None, model.module_list[23]))  # head与fpn的2个连接处
    fusion_modules[3].append((None, model.module_list[24]))

    for idx, module in enumerate(model.module_list):
        if 1 <= idx <= 11:  # backbone
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[0].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[0].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[0].append((None, current_module))

        elif 15 <= idx <= 22: # fpn
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[1].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[1].append((idx, current_module[-4], current_module[-3]))
            else:
                if not module.__str__().startswith('Zero') and module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[1].append((None, current_module))

        elif 25 <= idx <= 29:  # head26
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[2].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[2].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[2].append((None, current_module))
        elif 30 <= idx <= 34:  # head13
            if module.__str__().startswith('IRFBlock'):
                current_module = [m for m in module.children()]
                fusion_modules[3].append((module.use_res_connect, current_module[-2]))
                fusion_modules_in_block[3].append((idx, current_module[-4], current_module[-3]))
            else:
                if module.moduleList:
                    current_module = module.moduleList[0]
                    fusion_modules[3].append((None, current_module))

    def sychronBN(*bn):
        sum = torch.zeros_like(bn[0].weight.data)
        count = 0
        for bni in bn:
            sum += bni.weight.data.abs()
            count += 1
        for bni in bn:
            bni.weight.data = sum / count

    # 同步BN前
    # print(fusion_modules[0])
    # print(fusion_modules[0][5][1][1].weight.data)
    # print(fusion_modules[0][6][1][1].weight.data)
    for i in range(len(fusion_modules[0]) - 1, 0, -1):  # backbone 倒叙判断
        current = fusion_modules[0][i]  # 第i个元组
        prev = fusion_modules[0][i - 1]
        if current[0]:
            if prev[0]:  # 如果前一个也是res则三个同步
                sychronBN(current[1][1], prev[1][1], fusion_modules[0][i - 2][1])
            else:
                sychronBN(current[1][1], prev[1][1])
    # 同步BN后
    # print(fusion_modules[0][5][1][1].weight.data)
    # print(fusion_modules[0][6][1][1].weight.data)

    # 对于fpn直接全部同步
    sychronBN(*[m[1][1] for m in fusion_modules[1]])

    # 对于head直接全部同步
    sychronBN(*[m[1][1] for m in fusion_modules[2]])
    sychronBN(*[m[1][1] for m in fusion_modules[3]])
    
    # 对于IRFBlock前两层同步
    for modules in fusion_modules_in_block:
        for m in modules:
            sychronBN(m[1][1], m[2][1])


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def get_input_mask(idx, filters_mask):
    """
    返回上一层的mask
    """

    if idx == 0:
        return [np.ones(3)]

    if 2 <= idx <= 12: # fpn之前
        return filters_mask[(idx - 2) // 2]
    
    elif idx == 13 or idx == 14:
        return filters_mask[idx - 7]
    
    elif idx == 16:
        return [np.concatenate([filters_mask[idx - 12][-1], filters_mask[idx - 8][-1]])]
    
    elif idx == 17:
        return filters_mask[idx - 10]
    
    elif idx == 18 or idx == 19:
        return filters_mask[idx - 9]
        

def load_weights_from_loose_model(compact_model, loose_model, filters_mask):
    for idx, module in enumerate(loose_model.module_list):
        for m in module.modules():
            if isinstance(m, ConvBNRelu):
                if 2 <= idx <= 17:
                    filters = [mask[1] for mask in filters_mask if mask[0] == idx]
                    out_channel_idx = [np.argwhere(f)[:, 0].tolist() for f in filters][0]

                    input_mask = get_input_mask(idx, filters_mask)
                    # if idx == 16:
                    #     print(input_mask)

                    in_channel_idx = np.argwhere(input_mask[-1])[:, 0].tolist() # 确保只是最后一个的输出作为下层输入（pw输入）

                    # print("in ch id")
                    # if idx == 16:
                    #     print(in_channel_idx)
                    # return
                    for compact_m in compact_model.module_list[idx].modules():
                        if isinstance(compact_m, ConvBNRelu):
                            compact_conv = compact_m[0]
                            cpmpact_bn = compact_m[1]
                            
                    loose_conv, loose_bn = m[0], m[1]
                    
                    # if idx == 4:
                    #     print("before reloading")
                    #     print(compact_conv.weight.data.shape)
                    #     print("###############################")
                    #     print(loose_conv.weight.data.shape)

                    tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
                    compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    #                 if idx == 16:

    #                     print("after reloading")
    #                     print(compact_conv.weight.data[0][0])
    #                     print("###############################")
    #                     print(loose_conv.weight.data[0][0])

                    # Quant paras
                    compact_conv.quan_w_fn.s.data = loose_conv.quan_w_fn.s.data[out_channel_idx, :, :, :].clone()
                    compact_conv.quan_a_fn.s.data = loose_conv.quan_a_fn.s.data.clone()

                    # BN params
                    cpmpact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
                    cpmpact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
                    cpmpact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
                    cpmpact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

                else: # 其余无需剪枝层则按原样导入数据（1和最后层）
                    # BN paras
                    if len(compact_model.module_list[idx]) > 1:
                        compact_model.module_list[idx][1].weight.data = m[1].weight.data.clone()
                        compact_model.module_list[idx][1].bias.data = m[1].bias.data.clone()
                        compact_model.module_list[idx][1].running_mean.data = module[1].running_mean.data.clone()
                        compact_model.module_list[idx][1].running_var.data = module[1].running_var.data.clone()

                    # Conv paras
                    compact_conv, loose_conv = compact_model.module_list[idx][0], m[0]

                    if idx == 0:
                        compact_conv.weight.data = loose_conv.weight.data.clone()

                    elif idx == 18 or idx == 19:
                        input_mask = get_input_mask(idx, filters_mask)
                        in_channel_idx = np.argwhere(input_mask[-1])[:, 0].tolist() 
                        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
                        compact_conv.bias.data = loose_conv.bias.data.clone()

                    # Quant params
                    compact_conv.quan_w_fn.s.data = loose_conv.quan_w_fn.s.data.clone()
                    compact_conv.quan_a_fn.s.data = loose_conv.quan_a_fn.s.data.clone()

              

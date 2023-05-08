import timeit
import torch
from collections import OrderedDict
import gc
from building_blocks.builder import PRIMITIVES
from general_functions.utils import add_text_to_file, clear_files_in_the_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from thop import profile
# the settings from the page 4 of https://arxiv.org/pdf/1812.03443.pdf
#### table 2
CANDIDATE_BLOCKS_BACKBONE = ["CBL_k3", "CBL_k5", "CBL_k7",
                             "skip"]

CANDIDATE_BLOCKS_HEAD = ["CBL_k3", "CBL_k5", "CBL_k7", "skip"]

#### when calculate latency, use (16, 208, 208), (32, 104, 104), (64, 52, 52),
####                             (128, 26, 26), (256, 13, 13), (512, 13, 13)
SEARCH_SPACE_BACKBONE = OrderedDict([
    #### table 1. input shapes of 11 searched layers (considering with strides)
    # ("input_shape", [16, 32, 64,
    #                  128, 256, 512]),
    ("input_shape", [(16, 208, 208), (32, 104, 104), (64, 52, 52),
                     (128, 26, 26), (256, 13, 13), (512, 13, 13)]),
    # table 1. filter numbers over the 11 layers
    ("channel_size", [32, 64, 128,
                      256, 512, 1024]),
    # table 1. strides over the 11 layers
    ("strides", [1, 1, 1,
                 1, 1, 1])
])

SEARCH_SPACE_HEAD26 = OrderedDict([
    ("input_shape",
        [(384, 26, 26)]),
    ("channel_size",
        [256]),
    ("strides",
        [1])
])

SEARCH_SPACE_HEAD13 = OrderedDict([
    ("input_shape",
        [(256, 13, 13)]),
    ("channel_size",
        [512]),
    ("strides",
        [1])
])

YOLO_LAYER_26 = {
    'mask': (3,4,5),
    'anchors': (38,29,  71,50,  120,70,  152,119,  249,261,  262,132),
    'classes': 4,
    'num': 6,
    'jitter': .3,
    'ignore_thresh': .7,
    'truth_thresh': 1,
    'random': 1
}

YOLO_LAYER_13 = {
    'mask': (0,1,2),
    'anchors': (38,29,  71,50,  120,70,  152,119,  249,261,  262,132),
    'classes': 4,
    'num': 6,
    'jitter': .3,
    'ignore_thresh': .7,
    'truth_thresh': 1,
    'random': 1
}


def extract_anchors(yolo_layer):
    anchor_idxs = [int(x) for x in yolo_layer["mask"]]  # [1 2 3]
    # Extract anchors
    anchors = [int(x) for x in yolo_layer["anchors"]]  # [10 14  23 27  37 58  81 82  135 169  344 319]
    anchors = [(anchors[i], anchors[i + 1]) for i in
               range(0, len(anchors), 2)]  # [(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)]
    anchors = [anchors[i] for i in anchor_idxs]  # [(23,27),(37,58),(81,82)]
    return anchors

# **** to recalculate latency use command:
# l_table = LookUpTable(calculate_latency=True, path_to_file='lookup_table.txt', cnt_of_runs=50)
# results will be written to './supernet_functions/lookup_table.txt''
# **** to read latency from the another file use command:
# l_table = LookUpTable(calculate_latency=False, path_to_file='lookup_table.txt')
class LookUpTable:
    def __init__(self, candidate_blocks_backbone=None,
                 candidate_blocks_head=None,

                 search_space_backbone=None,
                 search_space_head26=None,
                 search_space_head13=None,

                 calculate_latency=False,
                 calculate_num_ops=False):

        if candidate_blocks_backbone is None:
            candidate_blocks_backbone = CANDIDATE_BLOCKS_BACKBONE
        if candidate_blocks_head is None:
            candidate_blocks_head = CANDIDATE_BLOCKS_HEAD


        if search_space_backbone is None:
            search_space_backbone = SEARCH_SPACE_BACKBONE
        if search_space_head26 is None:
            search_space_head26 = SEARCH_SPACE_HEAD26
        if search_space_head13 is None:
            search_space_head13 = SEARCH_SPACE_HEAD13


        self.num_anchors = len(YOLO_LAYER_26['mask'])
        self.num_cls = YOLO_LAYER_26['classes']
        self.cnt_layers = len(search_space_backbone["input_shape"])  # num. of layers for backbone
        self.head_layers = len(search_space_head26["input_shape"])  # num. of layers for head

        self.anchors_26 = extract_anchors(YOLO_LAYER_26)
        self.anchors_13 = extract_anchors(YOLO_LAYER_13)

        # constructors for each operation
        # select operations subset from PRIMITIVES
        self.lookup_table_operations = {op_name: PRIMITIVES[op_name] for op_name in candidate_blocks_backbone}
        self.lookup_table_operations_head = {op_name: PRIMITIVES[op_name] for op_name in candidate_blocks_head}


        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space_backbone)
        self.layers_parameters_head26, self.layers_input_shapes_head26 = self._generate_layers_parameters(
            search_space_head26)
        self.layers_parameters_head13, self.layers_input_shapes_head13 = self._generate_layers_parameters(
            search_space_head13)


        # lookup_table
        self.lookup_table_latency = None
        self.lookup_table_latency_head26 = None
        self.lookup_table_latency_head13 = None


        # lookup_table of num ops
        self.lookup_table_num_ops = None
        self.lookup_table_num_ops_head26 = None
        self.lookup_table_num_ops_head13 = None


        if calculate_latency:
            self._create_from_operations(cnt_of_runs=CONFIG_SUPERNET['lookup_table']['number_of_runs'],
                                         write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'],
                                         write_to_file_head26=CONFIG_SUPERNET['lookup_table'][
                                             'path_to_lookup_table_head26'],
                                         write_to_file_head13=CONFIG_SUPERNET['lookup_table'][
                                             'path_to_lookup_table_head13'],
                                         )
        else:
            self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'],
                                   path_to_file_head26=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_head26'],
                                   path_to_file_head13=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_head13'],
                                   )

        if calculate_num_ops:
            self._creat_num_ops_tables(write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_num_ops_table'],
                                       write_to_file_head26=CONFIG_SUPERNET['lookup_table']['path_to_num_ops_head26'],
                                       write_to_file_head13=CONFIG_SUPERNET['lookup_table']['path_to_num_ops_head13'],
                                       )

    @staticmethod
    def _generate_layers_parameters(search_space, prune=False):
        # layers_parameters are : C_in, C_out, expansion, stride, prune
        if prune:
            layers_parameters = [(search_space["input_shape"][layer_id],  # C_in for layer id
                                  search_space["channel_size"][layer_id],  # C_out for layer id
                                  None,  # expansion for layer id
                                  search_space["strides"][layer_id],  # stride for layer id
                                  search_space["prune"][layer_id],  # mid_depth&mid for layer id
                                  ) for layer_id in range(len(search_space["input_shape"]))]
        else:
            layers_parameters = [(search_space["input_shape"][layer_id],  # C_in for layer id
                                  search_space["channel_size"][layer_id],  # C_out for layer id
                                  # 6, # expansion for layer id
                                  search_space["strides"][layer_id],  # stride for layer id
                                  None,
                                  ) for layer_id in range(len(search_space["input_shape"]))]

        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]

        return layers_parameters, layers_input_shapes

    # CNT_OP_RUNS us number of times to check latency (we will take average)
    def _create_from_operations(self, cnt_of_runs, write_to_file=None,
                                write_to_file_head26=None, write_to_file_head13=None):
        self.lookup_table_latency = self._calculate_latency(self.cnt_layers,
                                                            self.lookup_table_operations,
                                                            self.layers_parameters,
                                                            self.layers_input_shapes,
                                                            cnt_of_runs)
        self.lookup_table_latency_head26 = self._calculate_latency(self.head_layers,
                                                                   self.lookup_table_operations_head,
                                                                   self.layers_parameters_head26,
                                                                   self.layers_input_shapes_head26,
                                                                   cnt_of_runs)
        self.lookup_table_latency_head13 = self._calculate_latency(self.head_layers,
                                                                   self.lookup_table_operations_head,
                                                                   self.layers_parameters_head13,
                                                                   self.layers_input_shapes_head13,
                                                                   cnt_of_runs)

        if write_to_file is not None:
            self._write_lookup_table_to_file(write_to_file,
                                             self.lookup_table_operations,
                                             self.cnt_layers,
                                             self.lookup_table_latency)
        if write_to_file_head26 is not None:
            self._write_lookup_table_to_file(write_to_file_head26,
                                             self.lookup_table_operations_head,
                                             self.head_layers,
                                             self.lookup_table_latency_head26)
        if write_to_file_head13 is not None:
            self._write_lookup_table_to_file(write_to_file_head13,
                                             self.lookup_table_operations_head,
                                             self.head_layers,
                                             self.lookup_table_latency_head13)


    def _creat_num_ops_tables(self, write_to_file=None,
                              write_to_file_head26=None, write_to_file_head13=None):
        self.lookup_table_num_ops = self._calculate_num_ops(self.cnt_layers,
                                                            self.lookup_table_operations,
                                                            self.layers_parameters,
                                                            self.layers_input_shapes)
        self.lookup_table_num_ops_head26 = self._calculate_num_ops(self.head_layers,
                                                                   self.lookup_table_operations_head,
                                                                   self.layers_parameters_head26,
                                                                   self.layers_input_shapes_head26)
        self.lookup_table_num_ops_head13 = self._calculate_num_ops(self.head_layers,
                                                                   self.lookup_table_operations_head,
                                                                   self.layers_parameters_head13,
                                                                   self.layers_input_shapes_head13)

        if write_to_file is not None:
            self._write_lookup_table_to_file(write_to_file,
                                             self.lookup_table_operations,
                                             self.cnt_layers,
                                             self.lookup_table_num_ops)
        if write_to_file_head26 is not None:
            self._write_lookup_table_to_file(write_to_file_head26,
                                             self.lookup_table_operations_head,
                                             self.head_layers,
                                             self.lookup_table_num_ops_head26)
        if write_to_file_head13 is not None:
            self._write_lookup_table_to_file(write_to_file_head13,
                                             self.lookup_table_operations_head,
                                             self.head_layers,
                                             self.lookup_table_num_ops_head13)


    def _calculate_latency(self, layers, operations, layers_parameters, layers_input_shapes, cnt_of_runs):
        LATENCY_BATCH_SIZE = 1
        latency_table_layer_by_ops = [{} for i in range(layers)]

        for layer_id in range(layers):
            for op_name in operations:
                op = operations[op_name](*layers_parameters[layer_id])
                input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
                globals()['op'], globals()['input_sample'] = op, input_sample
                total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
                                           globals=globals(), number=cnt_of_runs)
                # measured in micro-second
                avg_time = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
                latency_table_layer_by_ops[layer_id][op_name] = float('%.3f' % avg_time)

        return latency_table_layer_by_ops

    def _calculate_num_ops(self, layers, operations, layers_parameters, layers_input_shapes):
        LATENCY_BATCH_SIZE = 1
        num_ops_table_layer_by_ops = [{} for i in range(layers)]

        for layer_id in range(layers):
            for op_name in operations:
                op = operations[op_name](*layers_parameters[layer_id])
                input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
                macs, params = profile(op, inputs=(input_sample,))
                num_ops_table_layer_by_ops[layer_id][op_name] = (macs, params)

        return num_ops_table_layer_by_ops

    def _write_lookup_table_to_file(self, path_to_file, operations, layers, latency):
        clear_files_in_the_list([path_to_file])
        ops = [op_name for op_name in operations]
        text = [op_name + " " for op_name in ops[:-1]]
        text.append(ops[-1] + "\n")

        for layer_id in range(layers):
            for op_name in ops:
                text.append(str(latency[layer_id][op_name]))
                text.append(" ")
            text[-1] = "\n"
        text = text[:-1]

        text = ''.join(text)
        add_text_to_file(text, path_to_file)

    def _create_from_file(self, path_to_file, path_to_file_head26, path_to_file_head13):
        self.lookup_table_latency = self._read_lookup_table_from_file(path_to_file, self.cnt_layers)
        self.lookup_table_latency_head26 = self._read_lookup_table_from_file(path_to_file_head26, self.head_layers)
        self.lookup_table_latency_head13 = self._read_lookup_table_from_file(path_to_file_head13, self.head_layers)


    def _read_lookup_table_from_file(self, path_to_file, layers):
        latences = [line.strip('\n') for line in open(path_to_file)]
        ops_names = latences[0].split(" ")
        latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]

        lookup_table_latency = [{op_name: latences[i][op_id]
                                 for op_id, op_name in enumerate(ops_names)
                                 } for i in range(layers)]
        return lookup_table_latency

if __name__=="__main__":

    # lookup_table = LookUpTable(calculate_num_ops=True)

    label = [45,30,19,47,51,72,64,124,28,6,63,53,69,62,31,70,69,64,57,52,
             61,29,17,53,38,17,42,67,69,87,55,30,82,68,136,52,85,75,33,48]
    # yolo = [232,131,112,310,359,384,320,578,0,0,468,184,512,452,371,0,0,242,146,0,
    #         294,113,47,231,84,0,345,490,720,461,377,145,613,551,657,502,324,594,282,461]
    # err_y = []
    yolov3 = [190,0,74,287,381,284,0,533,89,0,451,197,492,422,356,430,0,147,74,268,314,
              148,70,240,132,498,218,447,539,453,278,265,644,415,640,371,337,610,216,443]
    err_yolov3 = []
    # ours = [123,108,72,188,253,317,0,519,71,0,412,136,506,388,485,345,0,181,116,234,281,
    #         131,126,211,193,474,231,396,626,347,384,241,628,402,436,451,305,537,268,428]
    # err_o = []
    for l, y in zip(label, yolov3):
        err_yolov3.append(y - l)
    # for l, y in zip(label, yolo):
    #     err_y.append(y - l)
    # for l, o in zip(label, ours):
    #     err_o.append(o - l)
    print(err_yolov3)
    # print(err_y)
    # print(err_o)
    pos_err_yolov3 = 0
    neg_err_yolov3 = 0
    for e in err_yolov3:
        if e > 0:
            pos_err_yolov3 += e
        else:
            neg_err_yolov3 += e
    print("pos err y:", pos_err_yolov3)
    print("neg err y:", neg_err_yolov3)
    print("total err y:", pos_err_yolov3-neg_err_yolov3)
    # pos_err_y = 0
    # neg_err_y = 0
    # for e in err_y:
    #     if e > 0:
    #         pos_err_y += e
    #     else:
    #         neg_err_y += e
    # print("pos err y:", pos_err_y)
    # print("neg err y:", neg_err_y)
    # print("total err y:", pos_err_y-neg_err_y)
    # pos_err_o = 0
    # neg_err_o = 0
    # for e in err_o:
    #     if e > 0:
    #         pos_err_o += e
    #     else:
    #         neg_err_o += e
    # print("pos err o:", pos_err_o)
    # print("neg err o:", neg_err_o)
    # print("total err o:", pos_err_o - neg_err_o)
    # candidate_blocks_backbone = CANDIDATE_BLOCKS_BACKBONE
    # operations = {op_name: PRIMITIVES[op_name] for op_name in candidate_blocks_backbone}
    # search_space_backbone = SEARCH_SPACE_BACKBONE
    #
    # layers_parameters, layers_input_shapes = LookUpTable._generate_layers_parameters(search_space_backbone)
    # for op_name in operations:
    #     op = operations[op_name](*layers_parameters[0])
    #     input_sample = torch.randn((1, *layers_input_shapes[0]))
    #     macs, params = profile(op, inputs=(input_sample,))
    #     # print((macs, params))
    #     print(op)




    # latences = [line.strip('\n') for line in open("./lookup_table.txt")]
    # print(latences)
    #
    # ops_names = latences[0].split(" ")
    # print(ops_names)
    # print("length opnames:", ops_names)
    # latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]
    # print(latences)
    # lookup_table_latency = [{op_name: latences[i][op_id]
    #                          for op_id, op_name in enumerate(ops_names)
    #                          } for i in range(11)]
    # print(lookup_table_latency)
    #
    # # import netron
    # # netron.start('./model.onnx')
    #
    # layers_parameters, layers_input_shapes = LookUpTable()._generate_layers_parameters(SEARCH_SPACE_BACKBONE)
    # operations = LookUpTable().lookup_table_operations
    # #print(operations)
    #
    # num_of_runs = 50
    #
    # for layer_id in range(11):
    #     for op_name in operations:
    #         op = operations[op_name](*layers_parameters[layer_id])
    #         #print(op)
    #         input_sample = torch.randn((1, *layers_input_shapes[layer_id]))
    #
    #         print(input_sample.shape)
    #
    #         globals()['op'], globals()['input_sample'] = op, input_sample
    #         total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
    #                                    globals=globals(), number=num_of_runs)
    #         break
    #     break
    # latency_table_layer_by_ops = total_time / num_of_runs * 1e6
    # print(latency_table_layer_by_ops)

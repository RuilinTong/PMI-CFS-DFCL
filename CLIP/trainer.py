import sys
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import gc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    if isinstance(seed_list, list):
        for seed in seed_list:
            args["seed"] = seed
            args["device"] = device
            _train(args)
    else:
        _train(args)


def compute_fgt(data):
    """
    Given a TxT data matrix, compute average forgetting at T-th task
    """
    num_tasks = len(data)
    output_matrix = np.zeros((num_tasks, num_tasks), dtype=float)

    # Copy the values from the input matrix to the output matrix
    for i, row in enumerate(data):
        output_matrix[i, :len(row)] = row
    data = output_matrix
    T = num_tasks - 1
    if T == 0:
        return 0.0, []
    fgt = 0.0
    fgts = []
    for i in range(T):
        fgts.append(np.max(data[:T,i]) - data[T][i])

    avg_fgt = sum(fgts) / float(num_tasks - 1)
    return avg_fgt, fgts


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    
    logs_name = "logs/{}/{}/{}_{}/{}-{}/".format(args["model_name"],args["dataset"], init_cls, args['increment'], args["backbone_type"],
            args["pretrained_weight"])
    # logs_name = "logs/{}/{}/{}_{}/ViT-B-16-laion400m".format(args["model_name"],args["dataset"], init_cls, args['increment'], args["backbone_type"],
    #         args["pretrained_weight"])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
    #     args["model_name"],
    #     args["dataset"],
    #     init_cls,
    #     args["increment"],
    #     args["prefix"],
    #     args["seed"],
    #     args["backbone_type"],
    # )
    # logfilename = "logs/{}/{}/{}_{}/{}_seed{}_{}_batch{}_epoch{}_lr{}".format(
    #     args["model_name"],
    #     args["dataset"],
    #     init_cls,
    #     args["increment"],
    #     args["prefix"],
    #     args["seed"],
    #     args["backbone_type"],
    #     args['batch_size'],
    #     args['tuned_epoch'],
    #     args['init_lr']
    # )
    if 'zs' in args["model_name"]:
        logfilename = "logs/{}/{}/{}_{}/{}_seed{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"],
            args["backbone_type"],
            args['pretrained_weight'],
        )
    else:
        logfilename = "logs/{}/{}/{}_{}/{}-{}/mem{}_{}_seed{}_batch{}_epoch{}_lr{}_dv{}_lv{}_dt{}_ps{}_adiv{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["backbone_type"],
            args["pretrained_weight"],
            # args['n_ctx_vision'],
            args['memory_per_class'],
            args["prefix"],
            args["seed"],
            args['batch_size'],
            args['tuned_epoch'],
            args['init_lr'],
            args['prompt_depth_vision'],
            args['n_ctx_vision'],
            args['prompt_depth_text'],
            args['pool_size'],
            args['alpha_div']
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    
    t = datetime.now().strftime("%d-%B-%Y-%H:%M:%S")
    try:
        tb_logger = SummaryWriter(log_dir=f"runs/{args['dataset']}/{args['model_name']}/{args['backbone_type']}-{args['pretrained_weight']}/seed_{args['seed']}_batch{args['batch_size']}_ep{args['tuned_epoch']}_lr{args['init_lr']}_depth{args['prompt_depth_text']}/{t}")
    except:
        tb_logger = SummaryWriter(log_dir=f"runs/{args['dataset']}/{args['model_name']}/{args['backbone_type']}-{args['pretrained_weight']}/seed_{args['seed']}_batch{args['batch_size']}_ep{args['tuned_epoch']}_lr{args['init_lr']}/{t}")

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes  # update args
    args["nb_tasks"] = data_manager.nb_tasks
    
    model = factory.get_model(args["model_name"], args, data_manager)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    forgetting = []
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager, tb_logger=tb_logger)
        gc.collect()
        cnn_accy, nme_accy = model.eval_task(tb_logger=tb_logger)
        # cnn_accy_l, nme_accy_l = model.eval_task_labeled(tb_logger=tb_logger)
        gc.collect()
        model.after_task()
        gc.collect()
        # model.plot_tsne()
        # model.save_model()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
            forgetting.append(cnn_accy["by_task"])
            logging.info("Forgetting: {}".format(compute_fgt(forgetting)))

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            # logging.info("CNN: {}".format(cnn_accy_l))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            try:
                forgetting.append(cnn_accy["by_task"])
                logging.info("Forgetting: {}\n".format(compute_fgt(forgetting)))
            except:
                pass
            # print('Forgetting:', compute_fgt(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

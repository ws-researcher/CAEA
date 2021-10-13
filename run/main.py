import time

import sys
sys.path.append("/home/ws/Workplace/KGAlign/")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from KGAlignModel.KGAlign import kgalign
from modules.args.args_hander import load_args
from modules.load.kgs import read_reversed_kgs_from_folder
# import tensorflow as tf
# print(tf.config.list_physical_devices("GPU"))

class ModelFamily(object):
    kgalign = kgalign


def get_model(model_name):
    return getattr(ModelFamily, model_name)

if __name__ == '__main__':
    t = time.time()
    args = load_args("./args/KGAlign_args_15K.json")

    kgs = read_reversed_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                                        remove_unlinked=False, complete_link=False)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.initModel()
    model.train()
    model.test()
    # model.save()
    # print("Total run time = {:.3f} s.".format(time.time() - t))
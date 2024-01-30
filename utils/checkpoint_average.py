from collections import OrderedDict
import os
import torch


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


def average_checkpoints(ckpt_path_list):
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(ckpt_path_list)
    for fpath in ckpt_path_list:
        print("Load ckpt from {}".format(fpath))
        state = load_checkpoint(path=fpath, use_cuda=True)
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model_state"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys

        for k in params_keys:
            p = model_params[k]
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model_state"] = averaged_params

    return new_state

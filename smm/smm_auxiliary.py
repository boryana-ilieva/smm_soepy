import collections
import yaml

def update_optim_paras(init_file_name, optim_paras):
    """Update the parameters with values of 
    the current optimizier iteration"""

    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)
    
    init_dict["PARAMETERS"]["optim_paras"] = [optim_paras.item(0),
                                              optim_paras.item(1),
                                              optim_paras.item(2),
                                              optim_paras.item(3),
                                              optim_paras.item(4),
                                              optim_paras.item(5),
                                              optim_paras.item(6),
                                              optim_paras.item(7),
                                              optim_paras.item(8),
                                              optim_paras.item(9),
                                              optim_paras.item(10),
                                              optim_paras.item(11),
                                              optim_paras.item(12),
                                              optim_paras.item(13),
                                              optim_paras.item(14),
                                              optim_paras.item(15),
                                              optim_paras.item(16),
                                              ]

    print_dict(init_dict)
    

def print_dict(init_dict, file_name="smm_init_file"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "PARAMETERS",
    ]
    for key_ in order:
        ordered_dict[key_] = init_dict[key_]

    with open("{}.soepy.yml".format(file_name), "w") as outfile:
        yaml.dump(ordered_dict, outfile, explicit_start=True, indent=4)
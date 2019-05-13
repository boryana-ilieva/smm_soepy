import collections
import yaml


def update_optim_paras(init_file_name, optim_paras):
    """Update the parameters with values of 
    the current optimizer iteration"""

    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    init_dict["PARAMETERS"]["gamma_0s1"] = optim_paras.item(0)
    init_dict["PARAMETERS"]["gamma_0s2"] = optim_paras.item(1)
    init_dict["PARAMETERS"]["gamma_0s3"] = optim_paras.item(2)
    init_dict["PARAMETERS"]["gamma_1s1"] = optim_paras.item(3)
    init_dict["PARAMETERS"]["gamma_1s2"] = optim_paras.item(4)
    init_dict["PARAMETERS"]["gamma_1s3"] = optim_paras.item(5)
    init_dict["PARAMETERS"]["g_s1"] = optim_paras.item(6)
    init_dict["PARAMETERS"]["g_s2"] = optim_paras.item(7)
    init_dict["PARAMETERS"]["g_s3"] = optim_paras.item(8)
    init_dict["PARAMETERS"]["delta_s1"] = optim_paras.item(9)
    init_dict["PARAMETERS"]["delta_s2"] = optim_paras.item(10)
    init_dict["PARAMETERS"]["delta_s3"] = optim_paras.item(11)
    init_dict["PARAMETERS"]["theta_p"] = optim_paras.item(12)
    init_dict["PARAMETERS"]["theta_f"] = optim_paras.item(13)
    init_dict["PARAMETERS"]["sigma_1"] = optim_paras.item(14)
    init_dict["PARAMETERS"]["sigma_2"] = optim_paras.item(15)
    init_dict["PARAMETERS"]["sigma_3"] = optim_paras.item(16)

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

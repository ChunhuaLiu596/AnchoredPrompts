import os, sys 
import configparser
import yaml 
import pandas as pd
from IPython.display import display
with open('config/clsb.yaml') as f :
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)


import argparse
# class ParseKwargs(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         setattr(namespace, self.dest, dict())
#         for value in values:
#             key, value = value.split('=')
#             getattr(namespace, self.dest)[key] = value

# parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
# args = parser.parse_args()

# parser = argparse.ArgumentParser()
# parser.add_argument("--filter_anchors_flag", action="store_true",  help="using raw anchors or filterred anchors")
# parser.add_argument("--filter_objects_flag", action="store_true",  help="filter the model predictions obj_label or not")
# parser.add_argument("--add_cpt_score", action="store_true", help="adding concept positioning test to filter anchors")
# parser.add_argument("--add_wordnet_path_score", action="store_true", help="adding concept positioning test to filter anchors")
# args = parser.parse_args()
# args = vars(args)

# def add_argument_to_config(args, config):
#     for name in  args.keys():
#         if name in config:
#             config[name] = args[name]
#             print(name)
#     return config 
# config =add_argument_to_config(args, config) 




from tabulate import tabulate, simple_separated_format

data = [['United States', 10, 12], ['United Kingdom', 15, 25], ['France', 14, 18]]
df = pd.DataFrame(data, columns = ['Country', 'Number1', 'Number2'])
df = df.reset_index(drop=True)
print(tabulate(df, tablefmt='latex').replace("&", "\t").replace("\\", ""))
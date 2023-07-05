import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="debug or not")
parser.add_argument("--debug", action="store_true",  help="debug or not")
parser.add_argument("--data_dir", default=None, help="debug or not")
parser.add_argument("--sub_col", default=None, help="which col is used as input x, sigular or plural")
parser.add_argument("--filter_anchors_flag", action="store_true",  help="using raw anchors or filterred anchors")
parser.add_argument("--filter_objects_flag", action="store_true",  help="filter the model predictions obj_label or not")
parser.add_argument("--filter_objects_with_anchors", action="store_true",  help="filter the model predictions obj_label or not")
parser.add_argument("--filter_objects_with_input", action="store_true",  help="filter the model predictions obj_label or not")
parser.add_argument("--add_cpt_score", action="store_true", help="adding concept positioning test to filter anchors")
parser.add_argument("--add_wordnet_path_score", action="store_true", help="adding concept positioning test to filter anchors")
parser.add_argument("--constrain_targets", action="store_true", help="constrain the target vocabulary of not")
parser.add_argument("--top_k", type=int, default=10, help="adding concept positioning test to filter anchors")
parser.add_argument("--top_k_anchors", type=int, default=10, help="how many anchors will be inserted into DAP ")
parser.add_argument('--max_anchor_num_list', nargs='+', type=int, default=[10], help= 'the max anchor num' )
# max_anchor_num_list
args = parser.parse_args()
args = vars(args)

print(args)
print(args['max_anchor_num_list'])
print(type(args['max_anchor_num_list']))
print(isinstance(args['max_anchor_num_list'], list))
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--rewiring', required=False, default='')
parser.add_argument('--num_iterations', type=int, required=False, default=0)
parser.add_argument('--layer_type', required=False, default='')
parser.add_argument('--gpu_index', required=True, default=0)
parser.add_argument('--num_seeds', type=int, required=False, default=10)
args = vars(parser.parse_args())

gpu_index = args['gpu_index']
num_seeds = args['num_seeds']
rewiring_methods = ['fosr', 'sdrf_orc', 'sdrf_bfc', 'none']
num_iterations = [3, 5, 10]
layer_types = ['GCN', 'GIN'] # 'R-GIN', 'R-GCN'

if(args['rewiring'] != ''):
    assert args['rewiring'] in rewiring_methods
    rewiring_methods = [args['rewiring']]
if(args['num_iterations'] != 0):
    num_iterations = [args['num_iterations']]
if(args['layer_type'] != ''):
    assert args['layer_type'] in layer_types
    layer_types = [args['layer_type']]

for rewiring in rewiring_methods:
    for num_iters in num_iterations:
        for layer_type in layer_types:
            cmd = f'python run_graph_classification.py --rewiring {rewiring} --num_iterations {num_iters} --layer_type {layer_type} --num_trials {num_seeds} --device cuda:{gpu_index}'
            print(cmd)

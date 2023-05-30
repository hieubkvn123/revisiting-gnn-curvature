
gpu_index = 0
layer_types = ["GCN", "GIN"]
batch_add = [1, 2, 3, 4, 5]
batch_remove = [0, 1, 2, 3]
num_iterations = [1, 2, 3, 4, 5]

cmd_template = """
python run_graph_classification.py 
    --rewiring borf 
    --layer_type {}
    --num_trials 100
    --device cuda:{}
    --borf_batch_add {}
    --borf_batch_remove {} 
    --num_iterations {}
"""

for layer in layer_types:
    for ba in batch_add:
        for br in batch_remove:
            for iters in num_iterations:
                cmd = cmd_template.format(layer, gpu_index, ba, br, iters)
                cmd = cmd.strip()
                cmd = cmd.replace('\n', '')
                cmd = cmd.replace('\t', ' ')

                print(cmd)

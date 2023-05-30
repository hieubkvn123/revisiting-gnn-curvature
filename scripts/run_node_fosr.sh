python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 150 --dataset cora
python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 100 --dataset citeseer
python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 50 --dataset texas
python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 125 --dataset cornell
python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 175 --dataset wisconsin
python run_node_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 50 --dataset chameleon

python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 50 --dataset cora
python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 200 --dataset citeseer
python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 150 --dataset texas
python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 75 --dataset cornell
python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 25 --dataset wisconsin
python run_node_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 25 --dataset chameleon

python run_graph_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 40 --dataset enzymes
python run_graph_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 5 --dataset imdb
python run_graph_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 10 --dataset mutag
python run_graph_classification.py     --rewiring fosr     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 30 --dataset proteins

python run_graph_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 5 --dataset enzymes
python run_graph_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 20 --dataset imdb
python run_graph_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 20 --dataset mutag
python run_graph_classification.py     --rewiring fosr     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 10 --dataset proteins

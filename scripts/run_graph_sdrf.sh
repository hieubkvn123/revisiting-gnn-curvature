python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 15 --dataset enzymes --sdrf_remove_edges
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 10 --dataset imdb --sdrf_remove_edges
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 20 --dataset mutag
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0    --num_iterations 5 --dataset proteins --sdrf_remove_edges

python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 5 --dataset enzymes --sdrf_remove_edges
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 10 --dataset imdb
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 10 --dataset mutag --sdrf_remove_edges
python run_graph_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0    --num_iterations 15 --dataset proteins --sdrf_remove_edges

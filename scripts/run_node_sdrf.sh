python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 12 --dataset cora  --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 175 --dataset citeseer
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 87 --dataset texas --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 100 --dataset cornell --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 25 --dataset wisconsin --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GCN    --num_trials 100    --device cuda:0     --num_iterations 50 --dataset chameleon --sdrf_remove_edges

python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 50 --dataset cora
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 25 --dataset citeseer
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 37 --dataset texas --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 25 --dataset cornell --sdrf_remove_edges
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 150 --dataset wisconsin
python run_node_classification.py     --rewiring sdrf_bfc     --layer_type GIN    --num_trials 100    --device cuda:0     --num_iterations 87 --dataset chameleon --sdrf_remove_edges

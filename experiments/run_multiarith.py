import asyncio
import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import GDesigner.agents
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONReader
from GDesigner.utils.globals import Time, Cost
from datasets.multiarith_dataset import multiarith_data_process, multiarith_get_predict, multiarith_check_correctness

from GDesigner.gdt.gtd_framework import GTDFramework
from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.llm_registry import get_llm_backend


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)
    return json.load(open(result_file, 'r', encoding='utf-8'))


def parse_args():
    parser = argparse.ArgumentParser(description="MultiArith experiments with HF backend")
    
    # Basic arguments
    parser.add_argument("--dataset_json", type=str, default="datasets/MultiArith/MultiArith_test.json")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-8B-Instruct")
    parser.add_argument('--mode', type=str, default='GTD', choices=['GTD'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--domain', type=str, default="multiarith")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    
    # GTD Mode Options
    gtd_group = parser.add_argument_group('GTD Mode Options')
    gtd_group.add_argument('--gtd_node_feat_dim', type=int, default=384)
    gtd_group.add_argument('--gtd_cond_dim', type=int, default=128)
    gtd_group.add_argument('--gtd_task_cond_input_dim', type=int, default=384)
    gtd_group.add_argument('--gtd_time_emb_dim', type=int, default=128)
    gtd_group.add_argument('--gtd_layers', type=int, default=2)
    gtd_group.add_argument('--gtd_heads', type=int, default=2)
    gtd_group.add_argument('--gtd_diffusion_steps', type=int, default=50)
    gtd_group.add_argument('--gtd_candidates', type=int, default=5)
    gtd_group.add_argument('--gtd-generate-data', action='store_true')
    gtd_group.add_argument('--gtd-train-models', action='store_true')
    gtd_group.add_argument('--gtd-datagen-limit', type=int, default=50)
    gtd_group.add_argument('--gtd-dataset-path', type=str, default='gtd_multiarith_dataset.jsonl')
    gtd_group.add_argument('--gtd-proxy-model-path', type=str, default='proxy_model_multiarith.pth')
    gtd_group.add_argument('--gtd-diffusion-model-path', type=str, default='diffusion_model_multiarith.pth')
    gtd_group.add_argument('--gtd-epochs', type=int, default=10)
    
    # GNN/MLP hyperparameters
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    
    # GTDFramework hyperparameters
    parser.add_argument('--time_embed_dim', type=int, default=128)
    parser.add_argument('--gt_num_layers', type=int, default=4)
    parser.add_argument('--gt_num_heads', type=int, default=8)
    
    args = parser.parse_args()
    return args


async def generate_initial_dataset(args, dataset):
    """Phase 1: Generate dataset with baseline topologies"""
    print(f"\n=== Phase 1: Generating dataset with {args.gtd_datagen_limit} samples ===")
    
    # Prepare agent list
    agent_names_list = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_nodes = len(agent_names_list)
    
    # Get agent embeddings - use GSM8K prompt set (MultiArith uses same math agents)
    prompt_set = PromptSetRegistry.get("gsm8k")
    agent_profiles = [prompt_set.get_description(name) for name in agent_names_list]
    node_features_base = [get_sentence_embedding(p) for p in agent_profiles]
    
    # Define baseline topologies
    def generate_static_topologies(n):
        fully_connected = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
        chain = [[0] * n for _ in range(n)]
        for i in range(n - 1):
            chain[i][i + 1] = 1
        return {
            'fully_connected': fully_connected,
            'chain': chain,
        }
    
    static_topologies = generate_static_topologies(num_nodes)
    
    generated_data = []
    proxy_data_list = []
    
    limit = min(args.gtd_datagen_limit, len(dataset))
    
    for i, record in enumerate(dataset[:limit]):
        task_query = record["task"]
        true_answer = record["answer"]
        task_condition_embedding = get_sentence_embedding(task_query)
        
        print(f"Processing {i+1}/{limit}: {task_query[:50]}...")
        
        for name, topology_matrix in static_topologies.items():
            print(f"  Topology: {name}")
            
            gdesigner_graph = Graph(
                "gsm8k",  # Use gsm8k domain for math tasks
                args.llm_name,
                agent_names_list,
                args.decision_method,
                fixed_spatial_masks=topology_matrix
            )
            
            raw_answer, _ = await gdesigner_graph.arun({"task": task_query}, args.num_rounds)
            
            predict_answer = multiarith_get_predict(raw_answer[0])
            is_solved = multiarith_check_correctness(predict_answer, true_answer)
            
            utility = 1.0 if is_solved else 0.0
            cost = sum(sum(row) for row in topology_matrix)
            
            # Store for dataset file
            generated_data.append({
                'graph': topology_matrix,
                'condition': task_condition_embedding.tolist(),
                'node_features': [f.tolist() for f in node_features_base],
                'performance': {'utility': utility, 'cost': cost}
            })
            
            # Store for proxy training
            adj_matrix = torch.tensor(topology_matrix, dtype=torch.float)
            edge_index, _ = dense_to_sparse(adj_matrix)
            proxy_data_list.append(Data(
                x=torch.tensor(node_features_base, dtype=torch.float),
                edge_index=edge_index,
                condition=torch.tensor(task_condition_embedding, dtype=torch.float).unsqueeze(0),
                true_rewards=torch.tensor([utility, cost], dtype=torch.float).unsqueeze(0)
            ))
            
            print(f"    Predicted: {predict_answer}, GT: {true_answer}, Solved: {is_solved}")
    
    # Save dataset
    with open(args.gtd_dataset_path, 'w') as f:
        for item in generated_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Phase 1 Complete: Saved {len(generated_data)} samples to {args.gtd_dataset_path}")


async def train_gtd_models(args, dataset):
    """Phase 2: Train proxy and diffusion models"""
    print(f"\n=== Phase 2: Training GTD models from {args.gtd_dataset_path} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    proxy_data_list = []
    diffusion_A0_list, diffusion_nodes_list, diffusion_cond_list = [], [], []
    
    with open(args.gtd_dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            rewards = item['performance']
            
            # Proxy data
            adj_matrix = torch.tensor(item['graph'], dtype=torch.float)
            edge_index, _ = dense_to_sparse(adj_matrix)
            proxy_data_list.append(Data(
                x=torch.tensor(item['node_features'], dtype=torch.float),
                edge_index=edge_index,
                condition=torch.tensor(item['condition'], dtype=torch.float).unsqueeze(0),
                true_rewards=torch.tensor([rewards['utility'], rewards['cost']], dtype=torch.float).unsqueeze(0)
            ))
            
            # Diffusion data (high quality only)
            if rewards['utility'] >= 0.0:
                diffusion_A0_list.append(item['graph'])
                diffusion_nodes_list.append(item['node_features'])
                diffusion_cond_list.append(item['condition'])
    
    print(f"Loaded {len(proxy_data_list)} samples for proxy, {len(diffusion_A0_list)} for diffusion")
    
    if not diffusion_A0_list:
        print("Warning: No high-quality graphs found, using all available data for training.")
        with open(args.gtd_dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                diffusion_A0_list.append(item['graph'])
                diffusion_nodes_list.append(item['node_features'])
                diffusion_cond_list.append(item['condition'])
    
    # Train Proxy Model
    print("\nTraining Proxy Model...")
    proxy_model = ProxyRewardModel(
        task_cond_input_dim=args.gtd_task_cond_input_dim,
        node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_reward_components=2
    ).to(device)
    
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    proxy_model.train()
    
    if proxy_data_list:
        for epoch in range(args.gtd_epochs):
            total_loss = 0
            for batch in PyGDataLoader(proxy_data_list, batch_size=16, shuffle=True):
                batch = batch.to(device)
                optimizer.zero_grad()
                pred_rewards = proxy_model(batch)
                loss = criterion(pred_rewards, batch.true_rewards)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{args.gtd_epochs}, Loss: {total_loss/len(proxy_data_list)*16:.4f}")
    
    torch.save(proxy_model.state_dict(), args.gtd_proxy_model_path)
    print(f"✓ Proxy model saved to {args.gtd_proxy_model_path}")
    
    # Train Diffusion Model
    print("\nTraining Diffusion Model...")
    gtd_framework = GTDFramework(
        task_cond_input_dim=args.gtd_task_cond_input_dim,
        node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim,
        time_embed_dim=args.time_embed_dim,
        gt_num_layers=args.gt_num_layers,
        gt_num_heads=args.gt_num_heads,
        device=device
    )
    
    if diffusion_A0_list:
        diffusion_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(diffusion_A0_list, dtype=torch.float),
                torch.tensor(diffusion_nodes_list, dtype=torch.float),
                torch.tensor(diffusion_cond_list, dtype=torch.float)
            ),
            batch_size=16,
            shuffle=True
        )
        gtd_framework.train_diffusion_model(dataloader=diffusion_dataloader, epochs=args.gtd_epochs)
    
    torch.save(gtd_framework.diffusion_model.state_dict(), args.gtd_diffusion_model_path)
    print(f"✓ Diffusion model saved to {args.gtd_diffusion_model_path}")


async def run_gtd_experiment(args, dataset):
    """Phase 3: Run inference with trained GTD"""
    print(f"\n=== Phase 3: GTD Inference on {len(dataset)} samples ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare agents
    agent_names_list = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_nodes = len(agent_names_list)
    
    # Load proxy model
    proxy_model = ProxyRewardModel(
        task_cond_input_dim=args.gtd_task_cond_input_dim,
        node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_reward_components=2
    )
    proxy_model.load_state_dict(torch.load(args.gtd_proxy_model_path))
    proxy_model.to(device).eval()
    proxy_model.reward_component_names = ['utility', 'cost']
    macp_weights = {'utility': 1.0, 'cost': -0.1}
    
    # Initialize GTD framework
    gtd_framework = GTDFramework(
        task_cond_input_dim=args.gtd_task_cond_input_dim,
        node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim,
        time_embed_dim=args.time_embed_dim,
        gt_num_layers=args.gt_num_layers,
        gt_num_heads=args.gt_num_heads,
        proxy_reward_model=proxy_model,
        macp_weights=macp_weights,
        device=device
    )
    gtd_framework.diffusion_model.load_state_dict(torch.load(args.gtd_diffusion_model_path))
    gtd_framework.diffusion_model.to(device).eval()
    
    # Get agent embeddings
    prompt_set = PromptSetRegistry.get("gsm8k")
    agent_profiles = [prompt_set.get_description(name) for name in agent_names_list]
    node_features_base = torch.tensor([get_sentence_embedding(p) for p in agent_profiles]).float().to(device)
    
    # Setup result file
    result_file = Path(GDesigner_ROOT, "result", f"gtd_{args.domain}", f"{args.llm_name.replace('/', '_')}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_solved = 0
    total_executed = 0
    
    for record in dataset:
        task_query = record["task"]
        true_answer = record["answer"]
        
        total_executed += 1
        print(f"\nSample {total_executed}/{len(dataset)}: {task_query[:50]}...")
        
        # Generate topology
        task_condition = torch.tensor(get_sentence_embedding(task_query)).float().unsqueeze(0).to(device)
        adj_matrix = (gtd_framework.generate_graphs(1, num_nodes, node_features_base.unsqueeze(0), task_condition, True).squeeze(0) > 0.5).int()
        
        # Run agents
        gdesigner_graph = Graph(
            "gsm8k",
            args.llm_name,
            agent_names_list,
            args.decision_method,
            fixed_spatial_masks=adj_matrix.tolist()
        )
        
        raw_answer, _ = await gdesigner_graph.arun({"task": task_query}, args.num_rounds)
        
        predict_answer = multiarith_get_predict(raw_answer[0])
        is_solved = multiarith_check_correctness(predict_answer, true_answer)
        total_solved += is_solved
        
        # Save results
        data = load_result(result_file)
        data.append({
            "Question": task_query,
            "Predicted": predict_answer,
            "Ground_Truth": true_answer,
            "Solved": is_solved,
            "Accuracy": total_solved / total_executed
        })
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"  Predicted: {predict_answer}, GT: {true_answer}, {'✓' if is_solved else '✗'}")
        print(f"  Running Accuracy: {total_solved/total_executed:.3f} ({total_solved}/{total_executed})")
    
    final_accuracy = total_solved / total_executed
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {final_accuracy:.3f} ({total_solved}/{total_executed})")
    print(f"{'='*60}")


async def main():
    args = parse_args()
    
    # Detect LLM backend
    backend = get_llm_backend(args.llm_name)
    print(f"Using LLM backend: {backend} for model: {args.llm_name}")
    
    # Load and process dataset
    dataset = JSONReader.parse_file(args.dataset_json)
    dataset = multiarith_data_process(dataset)
    print(f"Loaded {len(dataset)} samples from {args.dataset_json}")
    
    if args.gtd_generate_data:
        await generate_initial_dataset(args, dataset)
    elif args.gtd_train_models:
        await train_gtd_models(args, dataset)
    else:
        await run_gtd_experiment(args, dataset)


if __name__ == '__main__':
    asyncio.run(main())
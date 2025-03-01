
from typing import List, Tuple, Union
import torch
import logging
import argparse
import sys
from core.algorithms.load import init_evaluator
from core.demo.demo import Demo
from core.demo.load import init_demo
from core.resnet import ResNetConfig, TurboZeroResnet
from core.test.tester import  Tester
from core.test.tournament.tournament import Tournament, TournamentPlayer, load_tournament as load_tournament_checkpoint
from core.train.trainer import Trainer, init_history
from core.utils.checkpoint import load_checkpoint, load_model_and_optimizer_from_checkpoint
import matplotlib.pyplot as plt
import yaml

from envs.load import init_collector, init_env, init_tester, init_trainer

import numpy as np
from tqdm import trange

# def get_adj_matrix() -> torch.Tensor:
#     """
#     如果cos_matrix[i,j] == 0.5 or -0.5 就认为i和j是相邻的
#     """
#     batch_size = 1000
#     valid_cos_arr = torch.tensor([-1/2, 1/2], dtype=torch.float16, device="cuda", requires_grad=False)
#     vectors = torch.from_numpy(np.load("24D_196560_1.npy")).to("cuda").to(torch.float16)
#     adj_matrix = []
#     for i in trange(0, 196560, batch_size, desc="Generating Adjacency Matrix"):
#         end = min(i + batch_size, 196560)
#         vectors_a = vectors[i:end]
#         cos_arr = vectors_a @ vectors.T
#         m = cos_arr.shape[0]
#         adj_arr = torch.min(torch.abs(cos_arr[:, :, None] - valid_cos_arr), axis=2)[0] < 1e-6
#         adj_arr = adj_arr.cpu()
#         adj_matrix.append(adj_arr)
#     adj_matrix = torch.cat(adj_matrix, dim=0)
#     return adj_matrix

# ADJ_MATRIX = get_adj_matrix()

ADJ_MATRIX = None



# def get_adj_matrix() -> torch.Tensor:

#     batch_size = 1000
#     valid_cos_arr = torch.tensor([-1, -1/4, 1/4, 0, 1], dtype=torch.float16, device="cuda", requires_grad=False)
#     vectors = torch.from_numpy(np.load("24D_196560_1.npy")).to("cuda").to(torch.float16)
#     adj_matrix = []
#     for i in trange(0, 196560, batch_size, desc="Generating Adjacency Matrix"):
#         end = min(i + batch_size, 196560)
#         vectors_a = vectors[i:end]
#         cos_arr = vectors_a @ vectors.T
#         m = cos_arr.shape[0]
#         adj_arr = torch.min(torch.abs(cos_arr[:, :, None] - valid_cos_arr), axis=2)[0] < 1e-6
#         adj_arr = adj_arr.cpu()
#         adj_matrix.append(adj_arr)
#     adj_matrix = torch.cat(adj_matrix, dim=0)
#     return adj_matrix

# ADJ_MATRIX = get_adj_matrix()

# def get_adj_matrix() -> torch.Tensor:
#     num_gpus = torch.cuda.device_count()
#     total_size = 196560
#     batch_size = total_size // (num_gpus * 18)
#     chunk_size = total_size // num_gpus
#     valid_cos_arr = torch.tensor([-1, -1/4, 1/4, 0, 1], dtype=torch.float16, requires_grad=False)
#     vectors = torch.from_numpy(np.load("24D_196560_1.npy")).to(torch.float16)
    
#     # 创建CUDA流和存储结果的列表
#     streams = [torch.cuda.Stream(device=f'cuda:{i}') for i in range(num_gpus)]
#     gpu_results = [[] for _ in range(num_gpus)]
    
#     # 在每个GPU上启动异步计算
#     for gpu_id in range(num_gpus):
#         start_chunk = gpu_id * chunk_size
#         end_chunk = start_chunk + chunk_size if gpu_id < num_gpus - 1 else total_size
#         gpu_device = torch.device(f'cuda:{gpu_id}')
        
#         with torch.cuda.stream(streams[gpu_id]):
#             # 将数据移动到当前GPU
#             vectors_gpu = vectors.to(gpu_device)
#             valid_cos_arr_gpu = valid_cos_arr.to(gpu_device)
            
#             # 在当前GPU上处理分配的批次
#             for i in trange(start_chunk, end_chunk, batch_size, 
#                           desc=f"GPU {gpu_id} generating adjacency matrix"):
#                 end = min(i + batch_size, end_chunk)
#                 vectors_a = vectors_gpu[i:end]
                
#                 # 计算余弦相似度
#                 cos_arr = vectors_a @ vectors_gpu.T
                
#                 # 计算邻接关系
#                 adj_arr = torch.min(torch.abs(cos_arr[:, :, None] - valid_cos_arr_gpu), 
#                                   axis=2)[0] < 1e-6
                
#                 # 将结果保存到CPU
#                 gpu_results[gpu_id].append(adj_arr.cpu())
    
#     # 同步所有GPU
#     torch.cuda.synchronize()
    
#     # 收集并合并所有结果
#     all_results = []
#     for gpu_result in gpu_results:
#         all_results.extend(gpu_result)
    
#     # 按正确顺序合并结果
#     adj_matrix = torch.cat(all_results, dim=0)
#     return adj_matrix

# import time
# start_time = time.perf_counter()
# ADJ_MATRIX = get_adj_matrix()
# end_time = time.perf_counter()
# print(f"{end_time - start_time} for get_adj_matrix")

def setup_logging(logfile: str):
    if logfile:
        logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

def load_trainer_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    verbose_logging: bool = True,
    checkpoint: str = ''
) -> Trainer: 
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging,
        checkpoint=checkpoint
    )
    setup_logging(args.logfile)
    
    trainer = load_trainer(args, interactive=True)
    plt.close('all')
    return trainer

def load_tester_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    verbose_logging: bool = True,
    checkpoint: str = ''
) -> Tester:
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging,
        checkpoint=checkpoint
    )
    setup_logging(args.logfile)
    tester = load_tester(args, interactive=True)
    plt.close('all')
    return tester

def load_tournament_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    tournament_checkpoint: str = '',
    verbose_logging: bool = True
) -> Tuple[Tournament, List[dict]]:
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging,
        checkpoint=tournament_checkpoint
    )
    setup_logging(args.logfile)
    return load_tournament(args, interactive=True)

def load_demo_nb(
    config_file: str
) -> Demo:
    args = argparse.Namespace(
        config=config_file
    )
    return load_demo(args)
    

def load_config(config_file: str) -> dict:
    if config_file:
        with open(config_file, "r") as stream:
            raw_config = yaml.safe_load(stream)
    else:
        print('No config file provided, please provide a config file with --config')
        exit(1)
    return raw_config


def load_trainer(args, interactive: bool) -> Trainer:
    raw_config = load_config(args.config)

    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    episode_memory_device = torch.device('cpu') # we do not support episdoe memory on GPU yet

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        train_config = checkpoint['raw_train_config']
        env_config = checkpoint['raw_env_config']
    else:
        env_config = raw_config['env_config']
        train_config = raw_config['train_mode_config']
        
    run_tag = raw_config.get('run_tag', '')
    env_type = env_config['env_type']
    parallel_envs_train = train_config['parallel_envs']
    parallel_envs_test = train_config['test_config']['episodes_per_epoch']
    env_train = init_env(device, parallel_envs_train, env_config, args.debug, adj=ADJ_MATRIX)
    env_test = init_env(device, parallel_envs_test, env_config, args.debug, adj=ADJ_MATRIX)

    if args.checkpoint:
        model, optimizer = load_model_and_optimizer_from_checkpoint(checkpoint, env_train, device)
        history = checkpoint['history']
    else:
        model = TurboZeroResnet(ResNetConfig(**raw_config['model_config']), env_train.state_shape, env_train.policy_shape).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=raw_config['train_mode_config']['learning_rate'], momentum=raw_config['train_mode_config']['momentum'], weight_decay=raw_config['train_mode_config']['c_reg'])
        history = init_history()

    train_evaluator = init_evaluator(train_config['algo_config'], env_train, model)
    train_collector = init_collector(episode_memory_device, env_type, train_evaluator)
    test_evaluator = init_evaluator(train_config['test_config']['algo_config'], env_test, model)
    test_collector = init_collector(episode_memory_device, env_type, test_evaluator)
    tester = init_tester(train_config['test_config'], env_type, test_collector, model, history, optimizer, args.verbose, args.debug)
    trainer = init_trainer(device, env_type, train_collector, tester, model, optimizer, train_config, env_config, history, args.verbose, interactive, run_tag, debug=args.debug)
    return trainer

def load_tester(args, interactive: bool) -> Tester:
    raw_config = load_config(args.config)
    # test_config = raw_config['test_mode_config']
    test_config = raw_config['train_mode_config']['test_config']
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    episode_memory_device = torch.device('cpu')
    parallel_envs = test_config['episodes_per_epoch']
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        env_config = checkpoint['raw_env_config']
        env = init_env(device, parallel_envs, env_config, args.debug, adj=ADJ_MATRIX)
        model, _ = load_model_and_optimizer_from_checkpoint(checkpoint, env, device)
        history = checkpoint['history']

    else:
        print('No checkpoint provided, please provide a checkpoint with --checkpoint')
        exit(1)

    
    model = model.to(device)
    env_type = env_config['env_type']
    evaluator = init_evaluator(test_config['algo_config'], env, model)
    collector = init_collector(episode_memory_device, env_type, evaluator)
    tester = init_tester(test_config, env_type, collector, model, history, None, args.verbose, args.debug)
    return tester

def load_tournament(args, interactive: bool) -> Tuple[Tournament, List[dict]]:
    raw_config = load_config(args.config)
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    tournament_config = raw_config['tournament_mode_config']
    if args.checkpoint:
        tournament = load_tournament_checkpoint(args.checkpoint, device)
    else:
        env = init_env(device, tournament_config['num_games'], raw_config['env_config'], args.debug, adj=ADJ_MATRIX)
        tournament_name = tournament_config.get('tournament_name', 'tournament')
        tournament = Tournament(env, tournament_config['num_games'], tournament_config['num_tournaments'], device, tournament_name, args.debug)

    competitors = tournament_config['competitors']
        
    return tournament, competitors

def load_demo(args) -> Demo:
    raw_config = load_config(args.config)
    return init_demo(raw_config['env_config'], raw_config['demo_config'], torch.device('cpu'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TurboZero')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'test', 'demo', 'tournament'])
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logfile', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as stream:
            raw_config = yaml.safe_load(stream)
    else:
        print('No config file provided, please provide a config file with --config')
        exit(1)

    setup_logging(args.logfile)

    if args.mode == 'train':
        trainer = load_trainer(args, interactive=False)
        trainer.training_loop()
    elif args.mode == 'test':
        tester = load_tester(args, interactive=False)
        tester.collect_test_batch()
    elif args.mode == 'tournament':
        tournament, competitors = load_tournament(args, interactive=False)
        print(tournament.run(competitors, interactive=False))
    elif args.mode == 'demo':
        demo = load_demo(args)
        demo.run(print_state=True, interactive=False)

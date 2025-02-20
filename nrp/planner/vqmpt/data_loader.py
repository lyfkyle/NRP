'''Data loader for the 2D planner.
'''

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import skimage.io

import pickle
import re
import numpy as np
import open3d as o3d

import os
from os import path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import torch_geometric.data as tg_data

# from env.snake_8d.maze import Snake8DEnv
# env = Snake8DEnv(gui=False)
# env.load_occupancy_grid(np.zeros((100, 100)))

from env.fetch_11d.maze import Fetch11DEnv
env = Fetch11DEnv(gui=False)
env.load_occupancy_grid(np.zeros((150, 150, 20)))

q_min = np.array(env.robot.get_joint_lower_bounds())
q_max = np.array(env.robot.get_joint_higher_bounds())


class PathBiManipulationDataLoader(Dataset):
    ''' Loads each path for the bi-manipulation data
    '''

    def __init__(self, data_folder, env_list):
        '''
        :param data_folder: location of where file exists. 
        '''
        self.data_folder = data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                           for filei in os.listdir(osp.join(data_folder, f'env_{envNum:06d}'))
                           if filei.endswith('.p')
                           ]
        self.q_bi_max = np.c_[q_max, q_max]
        self.q_bi_min = np.c_[q_min, q_min]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        envNum, pathNum = self.index_dict[index]
        envFolder = osp.join(self.data_folder, f'env_{envNum:06d}')

        #  Load the path
        with open(osp.join(envFolder, f'path_{pathNum}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path['path']
        # Normalize the trajectory.
        q = (joint_path-self.q_bi_min)/(self.q_bi_max-self.q_bi_min)
        return {'path': torch.as_tensor(q)}


class PathManipulationDataLoader(Dataset):
    ''' Loads each path for the maniuplation data.
    '''

    def __init__(self, data_folder, env_list, num_joints=6, path_key='jointPath'):
        '''
        :param data_folder: location of where file exists. 
        '''
        self.data_folder = data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                        #    for filei in os.listdir(osp.join(data_folder, f'env_{envNum:06d}'))
                           for filei in os.listdir(osp.join(data_folder, f'env_{envNum}'))
                           if filei.endswith('.p')
                           ]
        self.num_joints = num_joints
        self.path_key = path_key

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        envNum, pathNum = self.index_dict[index]
        # envFolder = osp.join(self.data_folder, f'env_{envNum:06d}')
        envFolder = osp.join(self.data_folder, f'env_{envNum}')

        #  Load the path
        with open(osp.join(envFolder, f'path_{pathNum}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path[self.path_key]
        # Normalize the trajectory.
        q = (joint_path-q_min)/(q_max-q_min)
        return {'path': torch.as_tensor(q[:, :self.num_joints])}


def get_quant_manipulation_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various lengths.
    :param batch: the batch to consolidate
    '''
    data = {}
    # data['map'] = tg_data.Batch.from_data_list(
    #     [batch_i['map'] for batch_i in batch])
    data['map'] = torch.cat([batch_i['map'][None, :]
                            for batch_i in batch])
    data['input_seq'] = pad_sequence(
        [batch_i['input_seq'] for batch_i in batch], batch_first=True)
    data['target_seq_id'] = pad_sequence([batch_i['target_seq_id']
                                          for batch_i in batch], batch_first=True)
    data['length'] = torch.tensor(
        [batch_i['input_seq'].shape[0] for batch_i in batch])
    data['start_n_goal'] = torch.cat(
        [batch_i['start_n_goal'][None, :] for batch_i in batch])
    return data

import warnings

class QuantManipulationDataLoader(Dataset):
    ''' Data loader for quantized data values and associated point cloud.
    '''

    def __init__(self,
                 quantizer_model,
                 env_list,
                 map_data_folder,
                 quant_data_folder,
                 robot):
        '''
        :param quantizer_model: The quantizer model to use.
        :param env_list: List of environments to use for training.
        :param map_data_folder: location of the point cloud data.
        :param quant_data_folder: location of quantized data folder
        '''
        self.quant_data_folder = quant_data_folder
        self.map_data_folder = map_data_folder
        self.index_dict = []
        for envNum in env_list:
            # env_dir = osp.join(quant_data_folder, f'env_{envNum:06d}')
            env_dir = osp.join(quant_data_folder, f'env_{envNum}')
            if not osp.isdir(env_dir):
                continue
            for filei in os.listdir(env_dir):
                if filei.endswith('.p'):
                    self.index_dict.append((envNum, int(re.findall('[0-9]+', filei)[0])))
        if len(self.index_dict)==0:
            warnings.warn("No data found !!")

        self.quantizer_model = quantizer_model

        total_num_embedding = quantizer_model.embedding.weight.shape[0]
        self.start_index = total_num_embedding
        self.goal_index = total_num_embedding + 1
        self.robot = robot
        if robot=='14D':
            self.path_index = 'path'
            self.q_b_max = np.c_[q_max, q_max]
            self.q_b_min = np.c_[q_min, q_min]
        elif robot=='8D' or robot == '11D':
            self.path_index = 'path'
            self.q_b_max = q_max
            self.q_b_min = q_min
        elif robot=='6D':
            self.path_index = 'jointPath'
            self.q_b_max = q_max
            self.q_b_min = q_min

    def __len__(self):
        ''' Return the length of the dataset.
        '''
        return len(self.index_dict)*2

    def __getitem__(self, index):
        ''' Return the PC of the env and quant data.
        :param index: The index of the data.
        '''
        env_num, path_num = self.index_dict[index//2]

        # Load the occ grid data.
        # data_folder = osp.join(self.map_data_folder, f'env_{env_num:06d}')
        data_folder = osp.join(self.map_data_folder, f'env_{env_num}')
        if self.robot == "8D":
            occ_grid_file = osp.join(data_folder, 'occ_grid_small.txt')
            data_occ_grid = np.loadtxt(occ_grid_file).astype(np.uint8)
        elif self.robot == "11D":
            occ_grid_file = osp.join(data_folder, 'occ_grid_final.npy')
            with open(occ_grid_file, 'rb') as f:
                data_occ_grid = np.load(f)

        # Load start and goal states.
        with open(osp.join(data_folder, f'path_{path_num}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path[self.path_index]
            # flip array if index is odd.
            if index%2==1:
                joint_path = joint_path[::-1]
        
        # Normalize the trajectory.
        if self.robot=='14D':
            start_n_goal = ((joint_path-self.q_b_min)/(self.q_b_max-self.q_b_min))[[0, -1], :]
        elif self.robot=='6D':
            start_n_goal = ((joint_path-self.q_b_min)/(self.q_b_max-self.q_b_min))[[0, -1], :6]
        elif self.robot=='8D' or self.robot == '11D':
            start_n_goal = ((joint_path-self.q_b_min)/(self.q_b_max-self.q_b_min))[[0, -1], :]
        # Load the quant-data
        # with open(osp.join(self.quant_data_folder, f'env_{env_num:06d}', f'path_{path_num}.p'), 'rb') as f:
        with open(osp.join(self.quant_data_folder, f'env_{env_num}', f'path_{path_num}.p'), 'rb') as f:
            quant_data = pickle.load(f)
            # Flip array if index is odd.
            if index%2==1:
                quant_data['keys'] = quant_data['keys'][::-1].copy()

        with torch.no_grad():
            quant_vector = self.quantizer_model.embedding(
                torch.tensor(quant_data['keys']))
            quant_proj_vector = self.quantizer_model.output_linear_map(
                quant_vector)

        # add start vector:
        input_seq = torch.cat(
            [torch.ones(1, 512)*-1, quant_proj_vector], dim=0)
        input_seq_keys = np.r_[self.start_index,
                               quant_data['keys'], self.goal_index]

        return {
            # 'map': tg_data.Data(pos=torch.as_tensor(depth_points, dtype=torch.float)),
            'map': torch.as_tensor(data_occ_grid, dtype=torch.float).unsqueeze(0),
            'start_n_goal': torch.as_tensor(start_n_goal, dtype=torch.float),
            'input_seq': input_seq,
            'target_seq_id': torch.as_tensor(input_seq_keys[1:])
        }


def get_padded_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['path'] = pad_sequence(
        [batch_i['path'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['mask'] = pad_sequence([torch.ones(batch_i['path'].shape[0])
                                for batch_i in batch if batch_i is not None], batch_first=True)
    return data


class PathMixedDataLoader(Dataset):
    '''Loads each path, and extracts the masked positive and negative regions.
    The data is indexed in such a way that "hard" planning problems are equally distributed
    uniformly throughout the dataloading process.
    '''

    def __init__(self, envListMaze, dataFolderMaze, envListForest, dataFolderForest):
        '''
        :param envListMaze: The list of map environments to collect data from Maze.
        :param dataFolderMaze: The parent folder where the maze path files are located.
        :param envListForest: The list of map environments to collect data from Forest.
        :param dataFodlerForest: The parent folder where the forest path files are located.
            It should follow the following format:
                env1/path_0.p
                    ...
                env2/path_0.p
                    ...
                    ...
        '''
        assert isinstance(envListMaze, list), "Needs to be a list"
        assert isinstance(envListForest, list), "Needs to be a list"

        self.num_env = len(envListForest) + len(envListMaze)
        self.indexDictMaze = [('M', envNum, i)
                              for envNum in envListMaze
                              for i in range(len(os.listdir(osp.join(dataFolderMaze, f'env{envNum:06d}')))-1)
                              ]
        self.indexDictForest = [('F', envNum, i)
                                for envNum in envListForest
                                for i in range(len(os.listdir(osp.join(dataFolderForest, f'env{envNum:06d}')))-1)
                                ]
        self.dataFolder = {'F': dataFolderForest, 'M': dataFolderMaze}
        self.envList = {'F': envListForest, 'M': envListMaze}

    def __len__(self):
        return len(self.indexDictForest)+len(self.indexDictMaze)

    def __getitem__(self, idx):
        '''
        Returns the sample at index idx.
        returns dict: A dictonary of the encoded map and target points.
        '''
        DF, env, idx_sample = idx
        dataFolder = self.dataFolder[DF]
        mapEnvg = skimage.io.imread(
            osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png'), as_gray=True)

        with open(osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            data = pickle.load(f)

        if data['success']:
            path = data['path_interpolated']/24
            # path = data['path']/24
            return {
                'map': torch.as_tensor(mapEnvg),
                'path': torch.as_tensor(path)
            }


def get_quant_padded_sequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :]
                            for batch_i in batch])
    data['input_seq'] = pad_sequence(
        [batch_i['input_seq'] for batch_i in batch], batch_first=True)
    data['target_seq_id'] = pad_sequence([batch_i['target_seq_id']
                                          for batch_i in batch], batch_first=True)
    data['length'] = torch.tensor(
        [batch_i['input_seq'].shape[0] for batch_i in batch])
    data['start_n_goal'] = torch.cat(
        [batch_i['start_n_goal'][None, :] for batch_i in batch])
    return data


class QuantPathMixedDataLoader(Dataset):
    '''Loads the qunatized path.
    '''

    def __init__(
        self,
        quantizer_model,
        envListMaze,
        dataFolderMaze,
        quant_data_folder_maze,
        envListForest,
        dataFolderForest,
        quant_data_folder_forest
    ):
        '''
        :param envListMaze: The list of map environments to collect data from Maze.
        :param dataFolderMaze: The parent folder where the maze path files are located.
        :param envListForest: The list of map environments to collect data from Forest.
        :param dataFodlerForest: The parent folder where the forest path files are located.
            It should follow the following format:
                env1/path_0.p
                    ...
                env2/path_0.p
                    ...
                    ...
        '''
        assert isinstance(envListMaze, list), "Needs to be a list"
        assert isinstance(envListForest, list), "Needs to be a list"

        self.num_env = len(envListForest) + len(envListMaze)
        self.indexDictMaze = [('M', envNum, int(re.findall(r'\d+', f)[0]))
                              for envNum in envListMaze
                              for f in os.listdir(osp.join(quant_data_folder_maze, f'env{envNum:06d}')) if f[-2:] == '.p'
                              ]
        self.indexDictForest = [('F', envNum, int(re.findall(r'\d+', f)[0]))
                                for envNum in envListForest
                                for f in os.listdir(osp.join(quant_data_folder_forest, f'env{envNum:06d}')) if f[-2:] == '.p'
                                ]
        self.dataFolder = {'F': dataFolderForest, 'M': dataFolderMaze}
        self.quant_data_folder = {
            'F': quant_data_folder_forest, 'M': quant_data_folder_maze}
        self.quantizer_model = quantizer_model

        total_num_embedding = quantizer_model.embedding.weight.shape[0]
        self.start_index = total_num_embedding
        self.goal_index = total_num_embedding + 1

    def __len__(self):
        return len(self.indexDictForest)+len(self.indexDictMaze)

    def __getitem__(self, idx):
        '''
        Returns the sample at index idx.
        returns dict: A dictonary of the encoded map and target points.
        '''
        DF, env, idx_sample = idx
        dataFolder = self.dataFolder[DF]
        quant_data_folder = self.quant_data_folder[DF]

        map_env = skimage.io.imread(
            osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png'), as_gray=True)

        with open(osp.join(quant_data_folder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            quant_data = pickle.load(f)

        with open(osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            data = pickle.load(f)

        with torch.no_grad():
            quant_vector = self.quantizer_model.embedding(
                torch.tensor(quant_data['keys']))
            quant_proj_vector = self.quantizer_model.output_linear_map(
                quant_vector)

        # add start vector:
        input_seq = torch.cat(
            [torch.ones(1, 512)*-1, quant_proj_vector], dim=0)
        input_seq_keys = np.r_[self.start_index,
                               quant_data['keys'], self.goal_index]
        # Normalize the start and goal points
        start_n_goal = data['path'][[0, -1], :]/24
        return {
            'map': torch.as_tensor(map_env[None, :], dtype=torch.float),
            'start_n_goal': torch.as_tensor(start_n_goal, dtype=torch.float),
            'input_seq': input_seq,
            'target_seq_id': torch.as_tensor(input_seq_keys[1:])
        }

def get_mpnet_padded_seq(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['input_pos'] = pad_sequence(
        [batch_i['input_pos'] for batch_i in batch], batch_first=True
    )
    data['target_pos'] = pad_sequence(
        [batch_i['target_pos'] for batch_i in batch], batch_first=True
    )
    data['mask'] = pad_sequence([torch.ones(batch_i['input_pos'].shape[0])
                                for batch_i in batch], batch_first=True)
    data['env'] = torch.cat([batch_i['env'][None, :] for batch_i in batch], dim=0)
    return data

class MPNetDataLoader(Dataset):
    ''' Custom dataset object for training the MPNet model.
    '''

    def __init__(self, data_folder, env_list, max_point_cloud_size):
        '''
        :param data_folder: location of where file exists. 
        '''
        self.data_folder = data_folder
        self.index_dict = [(envNum, int(re.findall('[0-9]+', filei)[0]))
                           for envNum in env_list
                           for filei in os.listdir(osp.join(data_folder, f'env_{envNum:06d}'))
                           if filei.endswith('.p')
                           ]
        # Keeping env information fixed.
        self.max_point_cloud_size = max_point_cloud_size

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)*2

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        env_num, path_num = self.index_dict[index//2]
        envFolder = osp.join(self.data_folder, f'env_{env_num:06d}')
        
        # Load the pcd data.
        env_data_folder = osp.join(self.data_folder, f'env_{env_num:06d}')
        map_file = osp.join(env_data_folder, f'map_{env_num}.ply')
        data_PC = o3d.io.read_point_cloud(map_file, format='ply')
        total_number_points = np.array(data_PC.points).shape[0]
        ratio = min((1, (self.max_point_cloud_size+1)/total_number_points))
        down_sample_PC = data_PC.random_down_sample(ratio)
        # depth_points = np.array(data_PC.points)[:self.max_point_cloud_size, :]
        depth_points = np.array(down_sample_PC.points)[:self.max_point_cloud_size, :]
        #  Load the path
        with open(osp.join(envFolder, f'path_{path_num}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path['jointPath']
        # Normalize the trajectory.
        q = 2*(joint_path-q_min)/(q_max-q_min) - 1
        # set goal position.
        input_pos = np.concatenate([q[:-1, :6], np.ones_like(q[:-1, :6])*q[-1, :6]], axis=1)
        return {
            'input_pos': torch.as_tensor(input_pos, dtype=torch.float), 
            'target_pos': torch.as_tensor(q[1:, :6], dtype=torch.float), 
            'env': torch.as_tensor(depth_points.reshape(-1), dtype=torch.float)
        }
    
class MPNet14DDataLoader(Dataset):
    ''' Custom dataset object for training the MPNet model.
    '''

    def __init__(self, data_folder, env_list, max_point_cloud_size):
        '''
        :param data_folder: location of where file exists. 
        '''
        self.data_folder = data_folder
        self.index_dict = []
        for envNum in env_list:
            env_dir = osp.join(data_folder, f'env_{envNum:06d}')
            if not osp.isdir(env_dir):
                continue
            for filei in os.listdir(env_dir):
                if filei.endswith('.p'):
                    self.index_dict.append((envNum, int(re.findall('[0-9]+', filei)[0])))
        # Keeping env information fixed.
        self.max_point_cloud_size = max_point_cloud_size
        self.q_bi_max = np.c_[q_max, q_max]
        self.q_bi_min = np.c_[q_min, q_min]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.index_dict)*2

    def __getitem__(self, index):
        '''Gets the data item from a particular index.
        :param index: Index from which to extract the data.
        :returns: A dictionary with path.
        '''
        env_num, path_num = self.index_dict[index//2]
        envFolder = osp.join(self.data_folder, f'env_{env_num:06d}')
        
        # Load the pcd data.
        env_data_folder = osp.join(self.data_folder, f'env_{env_num:06d}')
        map_file = osp.join(env_data_folder, f'map_{env_num}.ply')
        data_PC = o3d.io.read_point_cloud(map_file, format='ply')
        total_number_points = np.array(data_PC.points).shape[0]
        ratio = min((1, (self.max_point_cloud_size+1)/total_number_points))
        down_sample_PC = data_PC.random_down_sample(ratio)
        # depth_points = np.array(data_PC.points)[:self.max_point_cloud_size, :]
        depth_points = np.array(down_sample_PC.points)[:self.max_point_cloud_size, :]
        #  Load the path
        with open(osp.join(envFolder, f'path_{path_num}.p'), 'rb') as f:
            data_path = pickle.load(f)
            joint_path = data_path['path']
        # Normalize the trajectory.
        q = 2*(joint_path-self.q_bi_min)/(self.q_bi_max-self.q_bi_min) - 1
        # set goal position.
        input_pos = np.concatenate([q[:-1, :], np.ones_like(q[:-1, :])*q[-1, :]], axis=1)
        return {
            'input_pos': torch.as_tensor(input_pos, dtype=torch.float), 
            'target_pos': torch.as_tensor(q[1:, :], dtype=torch.float),
            'env': torch.as_tensor(depth_points.reshape(-1), dtype=torch.float)
        }
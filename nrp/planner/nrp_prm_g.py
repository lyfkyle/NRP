import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import json
import time
import torch
import random
import math
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

CUR_DIR = osp.dirname(osp.abspath(__file__))

DEBUG = True


def dbg_print(x):
    if DEBUG:
        print(x)


MAX_NUM_RECUR = 1
LOCAL_ENV_SIZE = 2
NUM_OF_SAMPLES = 1
ADD_INTERMEDIATE_STATE = False


class LocalNeuralExtender8D:
    def __init__(self, env, dim, occ_grid_dim, model_path, device):
        self.env = env
        self.i = 0
        self.robot_dim = dim
        self.occ_grid_dim = occ_grid_dim
        self.device = device
        self.uniform_samples = None

        self._init_model(model_path)

        self.visualize = False
        self.print_time = False
        self.num_of_samples = NUM_OF_SAMPLES

    def _init_model(self, model_path):
        from planner.local_sampler_g.model_8d import VAEInference

        self.fk = self.env.utils.FkTorch(self.device)

        print("Loading checkpoint {}".format(model_path))

        # define networks
        z_dim = 5
        linkpos_dim = 12
        state_dim = self.robot_dim + linkpos_dim
        goal_state_dim = self.robot_dim + linkpos_dim + 1
        context_dim = state_dim + goal_state_dim
        self.generator = VAEInference(self.occ_grid_dim[1], z_dim, context_dim, state_dim)
        self.generator = torch.jit.script(self.generator)
        self.generator.load_state_dict(torch.load(model_path))
        self.generator.eval()
        print(self.device)
        self.generator.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(10):
            occ_grid_t = torch.zeros(
                (1, self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda"
            )
            start_t = torch.zeros((1, 20), device="cuda")
            goal_t = torch.zeros((1, 21), device="cuda")
            context_t = torch.cat((start_t, goal_t), dim=-1)
            self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)
        end_time = time.time()
        print("warmup takes {}".format(end_time - start_time))

    def set_robot_bounds(self, low, high):
        self.low = torch.tensor(low, device=self.device)
        self.high = torch.tensor(high, device=self.device)

    def enable_neural_select(self, enable):
        self.use_nueral_select = enable

    def set_uniform_samples(self, samples):
        self.uniform_samples = samples

    def set_mesh(self, mesh):
        """
        For visualization only
        """
        self.mesh = mesh

    def _preprocess_occ_grid(self, occ_grid_np):
        occ_grid_t = torch.tensor(occ_grid_np, device=self.device, dtype=torch.float).view(
            self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]
        )
        return occ_grid_t

    @torch.no_grad()
    def neural_expand(self, v, g, occ_grid_np):
        if self.print_time:
            start_time = time.perf_counter()

        # convert to GPU
        # num_to_sample = min(100, num_samples * 2)
        # num_to_sample = self.num_of_samples
        # z = torch.randn(num_to_sample, self.z_dim, device=device) # generate more samples first, since we prune later
        start = torch.tensor(v, device=self.device, dtype=torch.float)
        occ_grid_t = self._preprocess_occ_grid(occ_grid_np)
        goal = torch.tensor(g, device=self.device, dtype=torch.float)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: convert to GPU takes {}".format(end_time - start_time))

        # goal_in_env = False
        # if self.env.utils.is_robot_within_local_env(g, LOCAL_ENV_SIZE):
        #     goal_in_env = True

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: after random sampling takes {}".format(end_time - start_time))

        # select
        path = [v]
        tmp = torch.cat((start.view(1, -1), goal.view(1, -1)), dim=0)
        all_linkpos = self.fk.get_link_positions(tmp)
        start_linkpos = all_linkpos[0].view(-1)
        goal_linkpos = all_linkpos[1].view(-1)
        start_t = torch.cat((start, start_linkpos))
        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: forward kinematics takes {}".format(end_time - start_time))

        goal_direction = torch.atan2(goal[1], goal[0]).view(1)
        goal_t = torch.cat((goal, goal_linkpos, goal_direction))
        context_t = torch.cat((start_t, goal_t), dim=-1)
        samples = self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)[:, : self.robot_dim]

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            dbg_print("MyLocalSampler: calling selector takes {}".format(end_time - start_time))

        # get best samples
        # start_time  = time.time()
        selected_sample = samples[0].cpu().numpy().tolist()
        # torch.cuda.synchronize() # wait for mm to finish
        # end_time = time.perf_counter()
        # dbg_print("MyLocalSampler: convert back to cpu takes {}".format(end_time - start_time))

        if self.visualize:
            self.env.utils.visualize_nodes_local(
                occ_grid_np,
                [np.array(selected_sample)],
                v,
                g,
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, "selected_samples_viz_{}.png".format(self.i)),
            )

        path.append(selected_sample)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            dbg_print("MyLocalSampler: before return takes {}".format(end_time - start_time))

        self.i += 1

        return path


class LocalNeuralExtender11D(LocalNeuralExtender8D):
    def _init_model(self, model_path):
        from planner.local_sampler_g.model_11d import GenerativeSampler
        from env.fetch_11d.fk.model import ProxyFkTorch

        linkpos_dim = 24
        cur_dir = osp.dirname(osp.abspath(__file__))
        fkmodel_path = osp.join(cur_dir, "../env/fetch_11d/fk/models/model_fk.pt")
        self.fk = ProxyFkTorch(self.robot_dim, linkpos_dim, fkmodel_path, self.device)

        # selector
        print("Loading checkpoint {}".format(model_path))

        # define networks
        z_dim = 8
        state_dim = self.robot_dim + linkpos_dim
        goal_state_dim = self.robot_dim + linkpos_dim + 1
        context_dim = state_dim + goal_state_dim
        self.generator = GenerativeSampler(z_dim, context_dim, state_dim)
        self.generator = torch.jit.script(self.generator)
        self.generator.load_state_dict(torch.load(model_path))
        self.generator.eval()
        print(self.device)
        self.generator.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(10):
            occ_grid = torch.zeros((self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda")
            occ_grid_t = self.env.utils.add_pos_channels(occ_grid).unsqueeze(0)
            start_t = torch.zeros((1, 35), device="cuda")
            goal_t = torch.zeros((1, 36), device="cuda")
            context_t = torch.cat((start_t, goal_t), dim=-1)
            self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)
        end_time = time.time()
        print("warmup takes {}".format(end_time - start_time))

    def set_mesh(self, mesh):
        """
        For visualization only
        """
        self.mesh = mesh

    def _preprocess_occ_grid(self, occ_grid_np):
        occ_grid_t = torch.tensor(occ_grid_np, device=self.device, dtype=torch.float).view(
            self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]
        )
        occ_grid_t = self.env.utils.add_pos_channels(occ_grid_t)
        return occ_grid_t


class PRM:
    def __init__(
        self,
        sample_func,
        state_col_check_func,
        expand_func,
        distance_func,
        num_of_samples: int,
        num_neighbors: int,
    ):
        self._sample_func = sample_func
        self._state_col_check_func = state_col_check_func
        self._expand_func = expand_func
        self._distance_func = distance_func
        self.num_of_samples = num_of_samples
        self.num_neighbors = num_neighbors

    def clear(self):
        self.all_samples = []
        self.roadmap = nx.Graph()

    def compute_roadmap(self):
        """
        Offline computation of roadmap by sampling points.
        """
        while len(self.all_samples) < self.num_of_samples:
            collision_free = False
            while not collision_free:
                v = self._sample_func()[0]
                collision_free = self._state_col_check_func(v)

            self.all_samples.append(v)

        print("PRM/compute_roadmap: finished adding {} vertices".format(self.num_of_samples))
        # print(self.V)

        for v in tqdm(self.all_samples):
            neighbors = self.get_nearest_neighbors(self.all_samples, v, self.num_neighbors)
            # print("neighbours {}".format(neighbours))
            path = self._expand_func(v, neighbors)
            if len(path) == 3:
                self.roadmap.add_edge(tuple(v), tuple(path[1]))
                self.roadmap.add_edge(tuple(path[1]), tuple(path[2]))

        return self.roadmap

    def get_nearest_neighbors(self, all_vertices, v, n_neighbors: int = 1):
        """
        return the closest neighbors of v in all_vertices.

        Args:
            all_vertices (list[tuple]): a list of vertices
            v (tuple): the target vertex.
            n_neighbors (int): number of nearby neighbors.

        Returns:
            (list[tuple]): a list of nearby vertices
        """
        n_neighbors = min(n_neighbors, len(all_vertices))

        all_vertices = np.array(all_vertices)
        # v = np.array(v).reshape(1, -1)

        nbr_vertices = [n for n in all_vertices if math.fabs(n[0] - v[0]) < 2.0 and math.fabs(n[1] - v[1]) < 2.0]
        return nbr_vertices

        # nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm="ball_tree", metric=self._distance_func).fit(all_vertices)
        # distances, indices = nbrs.kneighbors(v)
        # # print("indices {}".format(indices))
        # nbr_vertices = np.take(np.array(all_vertices), indices.ravel(), axis=0).tolist()
        # nbr_vertices = [tuple(v) for v in nbr_vertices]
        # return nbr_vertices[1:]


class NRP_PRM_g:
    """
    Purely discriminative
    Use selector to extend
    """

    def __init__(self, env, model_path, dim=8, occ_grid_dim=[1, 40, 40], device=torch.device("cuda")):
        self.dim = dim
        self.env = env
        self.nueral_select_cnt = 1
        self.log_dir = None
        self.collect_dataset = False
        self.log_extension_info = False
        self.use_online_samples = False
        self.ego_local_env = True
        self.only_sample_col_free = False
        self.sample_around_goal = False

        if env.name == "snake_8d":
            self.sampler = LocalNeuralExtender8D(env, self.dim, occ_grid_dim, model_path, device=device)
        elif env.name == "fetch_11d":
            self.sampler = LocalNeuralExtender11D(env, self.dim, occ_grid_dim, model_path, device=device)

        self.algo = PRM(
            self.sample_uniform, self.state_col_checker, self.expand_fn, self.distance_fn, num_of_samples=10000, num_neighbors=10
        )
        self.add_intermediate_state = False

        self.sl_bias = 0

        random.seed(0)

    def clear(self):
        self.env = None
        self.global_occ_grid = None
        self.sample_memory = {}
        self.extension_col_cnt = 0
        self.algo.clear()
        self.neural_extend_time = 0
        self.neural_extend_success_time = 0
        self.neural_extend_fail_time = 0
        self.col_check_time = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0
        self.neural_extend_cnt = 0
        self.neural_select_cnt = 0
        self.neural_extend_col_cnt = 0
        self.selector_col_cnt = 0
        self.dataset_col_rate = 0
        self.dataset_diff_cnt = 0
        self.sampler.i = 0

    def solve(self, env, mesh=None):
        self.clear()
        self.algo.clear()
        self.env = env
        self.global_occ_grid = self.env.get_occupancy_grid()
        self.sampler.set_mesh(mesh)
        self.low_bounds = env.robot.get_joint_lower_bounds()
        self.high_bounds = env.robot.get_joint_higher_bounds()

        self.sampler.warmup()

        roadmap = self.algo.compute_roadmap()
        return roadmap

    def state_col_checker(self, v):
        return self.env.pb_ompl_interface.is_state_valid(v)

    def col_checker(self, v1, v2):
        # start_time = time.time()
        valid = self.env.utils.is_edge_free(self.env, v1, v2)
        # end_time = time.time()
        # print("collision check takes: {}".format(end_time - start_time))
        return valid

    def heuristic(self, v1, v2):
        return self.env.utils.calc_edge_len(v1, v2)

    def sample_uniform(self, num_samples=1):
        # print(self.algo.graph.number_of_nodes())
        # self.env.utils.visualize_nodes(self.global_occ_grid, list(self.algo.graph.nodes()), None, None, self.start, self.goal)

        if num_samples == 0:
            return []

        samples = []
        low = self.env.robot.get_joint_lower_bounds()
        high = self.env.robot.get_joint_higher_bounds()
        for _ in range(num_samples):
            if self.sample_around_goal:
                noises = np.random.normal(scale=0.2, size=len(self.goal))
                random_state = np.array(self.goal) + noises
                random_state = random_state.tolist()
                if self.only_sample_col_free:
                    while not self.env.pb_ompl_interface.is_state_valid(random_state):
                        noises = np.random.normal(size=len(self.goal))
                        random_state = np.array(self.goal) + noises
                        random_state = random_state.tolist()
            else:
                random_state = [0] * self.env.robot.num_dim
                for i in range(self.env.robot.num_dim):
                    random_state[i] = random.uniform(low[i], high[i])

                if self.only_sample_col_free:
                    while not self.env.pb_ompl_interface.is_state_valid(random_state):
                        random_state = [0] * self.env.robot.num_dim
                        for i in range(self.env.robot.num_dim):
                            random_state[i] = random.uniform(low[i], high[i])

            samples.append(random_state)

        return samples

    def distance_fn(self, v1, v2):
        return self.env.utils.calc_edge_len(v1, v2)

    def expand_fn(self, v, gs):
        paths = []
        if random.uniform(0, 1) < self.sl_bias:
            for g in gs:
                paths.append(self.sl_expand_fn(v, g))
            return paths
        else:
            paths = self.neural_expand_fn(v, gs)
            return paths

    def sl_expand_fn(self, v, g):
        start_time = time.time()
        sl_extend_path = self.expand_rrt(v, g)
        end_time = time.time()

        self.col_check_time += end_time - start_time
        if len(sl_extend_path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        if self.log_extension_info:
            if not np.allclose(np.array(sl_extend_path[-1]), np.array(g)):
                self.extension_col_cnt += 1

        # end_time = time.time()
        # print("sl_extend takes {}".format(end_time - start_time))

        return sl_extend_path

    def neural_expand_fn(self, v, gs):
        start_time = time.time()

        cur_v = v
        path = [v]
        for _ in range(1):
            recur_start_time = time.time()
            success = False

            local_gs = [self.env.utils.global_to_local(g, cur_v) for g in gs]
            local_g = self.env.utils.global_to_local(g, cur_v)
            local_v = self.env.utils.global_to_local(cur_v, cur_v)
            local_occ_grid = self.env.get_local_occ_grid(cur_v)

            # assert self.maze.pb_ompl_interface.is_state_valid(cur_v)
            local_path = self.sampler.neural_expand(local_v, local_g, local_occ_grid)
            global_path = [self.env.utils.local_to_global(x, cur_v) for x in local_path]

            if len(global_path) > 1:  # extend to a new place.
                if (
                    global_path[-1][0] > self.low_bounds[0]
                    and global_path[-1][0] < self.high_bounds[0]
                    and global_path[-1][1] > self.low_bounds[1]
                    and global_path[-1][1] < self.high_bounds[1]
                ):
                    path.append(global_path[-1])
                    success = True

                    # change cur_v if using ego view mode
                    if self.ego_local_env:
                        cur_v = global_path[-1]
                else:
                    print("Neural extension selects a samples outside env bounds. This should happen rarely")

            self.neural_select_cnt += 1
            recur_end_time = time.time()
            if success:
                self.neural_extend_success_time += recur_end_time - recur_start_time
            else:
                self.neural_extend_fail_time += recur_end_time - recur_start_time

        self.neural_extend_cnt += 1

        # print("Extension num: {}".format(self.algo.num_sampler_called))

        # extend towards global g.
        path.append(g)
        # print(len(path))

        end_time = time.time()
        self.neural_extend_time += end_time - start_time
        # print("Calling neural networks takes : {}".format(end_time - start_time))

        # Collision check
        start_time = time.time()
        final_path = [v]
        for i in range(1, len(path)):
            v1 = path[i - 1]
            v2 = path[i]
            res_path = self.expand_rrt(v1, v2)
            if len(res_path) > 1:
                final_path += res_path[1:]
            if not np.allclose(np.array(res_path[-1]), np.array(v2)):
                # print("collision detected!!!")
                break

        end_time = time.time()
        self.col_check_time += end_time - start_time
        # print("Collision check takes : {}".format(end_time - start_time))

        # print(final_path)

        # Avoid float precision issues
        if np.allclose(np.array(final_path[-1]), np.array(g)):
            final_path[-1] = g

        # record down information
        if self.log_extension_info:
            self.dump_extension_information(path, final_path, g)

        if len(final_path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        # print(final_path, path)

        return final_path, [], path

    def dump_extension_information(self, path, final_path, g):
        # self.env.utils.visualize_nodes_global(self.global_occ_grid, final_path, v, g)

        # calculate col rate
        if len(path) <= 1:
            selector_col = 0
        else:
            selector_col = 0
            for i in range(1, len(path) - 1):
                v1 = path[i - 1]
                v2 = path[i]
                is_free = self.env.utils.is_edge_free(self.env, v1, v2)
                if not is_free:
                    selector_col += 1

            # print(selector_col, len(path))
            self.selector_col_cnt += selector_col

        if not np.allclose(np.array(path[-1]), np.array(final_path[-1])):
            self.extension_col_cnt += 1
            self.neural_extend_col_cnt += 1

        extension_data = {}
        extension_data["local_planning_target"] = g
        extension_data["path_intended"] = path
        extension_data["path_actual"] = final_path

        if self.log_dir is not None:
            with open(osp.join(self.log_dir, "extension_data_{}.json".format(self.algo.num_extensions + 1)), "w") as f:
                json.dump(extension_data, f)

    def expand_rrt(self, v, g):
        return self.env.utils.rrt_extend(self.env, v, g, step_size=0.1)

    def convert_to_local_g(self, goal_pos):
        if math.fabs(goal_pos[0]) > 2.5 or math.fabs(goal_pos[1]) > 2.5:
            theta = math.atan2(goal_pos[1], goal_pos[0])
            if math.fabs(goal_pos[0]) > math.fabs(goal_pos[1]):
                tmp = 2.5 if goal_pos[0] > 0 else -2.5
                goal_pos[1] = math.tan(theta) * tmp
                goal_pos[0] = tmp
            else:
                tmp = 2.5 if goal_pos[1] > 0 else -2.5
                goal_pos[0] = tmp / math.tan(theta)
                goal_pos[1] = tmp

        return goal_pos[:2]

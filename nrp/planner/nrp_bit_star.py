# Adopted from https://github.com/rainorangelemon/gnn-motion-planning
import os.path as osp
import numpy as np
import math
import json
import heapq
import time
import networkx as nx
import torch
import itertools

from nrp.planner.local_sampler_g.local_sampler_g import LocalNeuralExpander11D, LocalNeuralExpander8D


INF = float("inf")
LOCAL_ENV_SIZE = 2.0


class NRPBITStar:
    def __init__(
        self,
        env,
        model_path,
        dim=11,
        occ_grid_dim=[40, 40, 20],
        device=torch.device("cuda"),
        batch_size=100,
        neural_sample_size=256,
        sampling=None,
        log=False,
    ):
        self.env = env
        self.dim = dim
        self.neural_select_cnt = 1
        self.ego_local_env = True
        self.max_neural_samples = neural_sample_size
        self.log = log

        if env.name == "snake_8d":
            self.sampler = LocalNeuralExpander8D(env, self.dim, occ_grid_dim, model_path, device=device)
        elif env.name == "fetch_11d":
            self.sampler = LocalNeuralExpander11D(env, self.dim, occ_grid_dim, model_path, device=device)
            # self.sampler.print_time = True

        # Hyperparameters
        self.batch_size = batch_size

        if sampling is None:
            self.sampling = self.informed_sample
        else:
            self.sampling = sampling

        self.n_collision_points = 0
        self.n_free_points = 2

        self.clear()

        self.env_mesh_path = None
        self.env_occ_grid = None
        self.log_dir = None

    def col_checker_fn(self, v1, v2):
        valid = self.env.utils.is_edge_free(self.env, v1, v2)
        return valid

    def radius_init(self):
        from scipy import special

        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / (self.n_collision_points + self.n_free_points)
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample_init(self):
        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
        a_1 = (np.array(self.goal) - np.array(self.start)) / self.c_min
        id1_t = np.array([1.0] * self.dimension)
        M = np.dot(a_1.reshape((-1, 1)), id1_t.reshape((1, -1)))
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(
            np.dot(U, np.diag([1] * (self.dimension - 1) + [np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh
        )

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u**2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num):
        # print("sampling")
        # The following are exactly from informed RRT* paper
        if c_best < float("inf"):
            c_b = math.sqrt(c_best**2 - self.c_min**2) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
        sample_array = []
        cur_num = 0
        while cur_num < sample_num:
            if c_best < float("inf"):
                x_ball = self.sample_unit_ball()
                random_point = tuple(np.dot(np.dot(self.C, L), x_ball) + self.center_point)
            else:
                random_point = self.get_random_point()

            if self.is_point_free(random_point):
                sample_array.append(random_point)
            cur_num += 1

        return sample_array

    def get_random_point(self):
        point = self.bounds[:, 0] + np.random.random(self.dimension) * self.ranges
        return tuple(point)

    def is_point_free(self, point):
        if (np.array(point) < self.bounds[:, 0]).any() or (np.array(point) > self.bounds[:, 1]).any():
            result = False
        else:
            result = self.env.pb_ompl_interface.is_state_valid(point)

        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        # result = self.env._edge_fp(np.array(edge[0]), np.array(edge[1]))
        result = self.col_checker_fn(edge[0], edge[1])
        # self.T += self.env.k
        return result

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            self.num_edge_col += 1
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        # return np.linalg.norm(np.array(point1) - np.array(point2))
        return self.env.utils.calc_edge_len(point1, point2)
        # return self.distance_fn(point1, point2).item()

    def get_edge_value(self, edge):
        # sort value for edge
        return (
            self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1], self.goal)
        )

    def get_point_value(self, point):
        # sort value for point
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point)
                if self.graph.has_edge(point, parent):
                    self.graph.remove_edge(point, parent)

    def prune(self, c_best):
        # print("pruning")
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        vertices_to_remove = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                    vertices_to_remove.append(point)
                else:
                    vertices_temp.append(point)
            else:
                vertices_to_remove.append(point)

        self.vertices = vertices_temp
        self.graph.remove_nodes_from(vertices_to_remove)

    def expand_vertex(self, point):
        # print("expanding vertex")
        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = (
                self.heuristic_cost(self.start, point)
                + self.heuristic_cost(point, neighbor)
                + self.heuristic_cost(neighbor, self.goal)
            )
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)

            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = (
                        self.heuristic_cost(self.start, point)
                        + self.heuristic_cost(point, neighbor)
                        + self.heuristic_cost(neighbor, self.goal)
                    )
                    if estimated_f_score < self.g_scores[self.goal]:
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def clear(self):
        self.env = None
        self.start = None
        self.goal = None

        # This is the tree
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        self.graph = nx.Graph()
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()
        self.num_samples = 0
        self.num_sample_max = float("inf")
        self.num_col_check = 0
        self.max_edge_eval = float("inf")
        self.allowed_time = float("inf")

        self.r = INF
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        self.col_check_time = 0
        self.num_edge_col = 0
        self.total_running_time = 0
        self.total_vertex_selection_time = 0
        self.total_vertex_extension_time = 0
        self.loop_cnt = 0

    def setup_planning(self):
        bounds = self.env.robot.get_joint_bounds()
        self.dimension = self.env.robot.num_dim
        self.bounds = np.array(bounds).reshape(-1, 2)

        low_bounds = self.env.robot.get_joint_lower_bounds()
        high_bounds = self.env.robot.get_joint_higher_bounds()
        low_bounds[0] = -2
        low_bounds[1] = -2
        high_bounds[0] = 2
        high_bounds[1] = 2
        self.sampler.set_robot_bounds(low_bounds, high_bounds)

        # print(self.bounds)
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        # print(self.ranges)

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # add goal to the samples
        self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        self.graph.add_node(self.goal)

        # add start to the tree
        self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        self.graph.add_node(self.start)

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        self.sampler.warmup()

        return radius_constant

    def solve(self, env, start, goal, allowed_time=float("inf"), max_samples=float("inf")):
        self.clear()
        self.env = env
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.max_edge_eval = max_samples
        self.allowed_time = allowed_time

        self.setup_planning()
        path = self.plan()

        return path

    def solve_step_expansion(self, env, start, goal, max_extensions, step_size=50):
        self.clear()
        self.env = env
        self.start = tuple(start)
        self.goal = tuple(goal)

        self.setup_planning()

        res = []
        i = 0
        for max_ext in range(step_size, max_extensions + 1, step_size):
            self.max_edge_eval = max_ext
            path = self.plan()
            success = len(path) > 0
            res.append((success, path))
            i += 1

        return res

    def solve_step_time(self, maze, start, goal, max_time, step_size, mesh=None):
        self.clear()
        self.env = maze
        self.start = tuple(start)
        self.goal = tuple(goal)

        self.setup_planning()

        res = []
        i = 0
        max_t = step_size
        self.start_time = time.time()
        while max_t <= max_time + 1e-4:
            self.allowed_time = self.start_time + max_t
            path = self.plan()

            success = len(path) > 0
            res.append((success, path))
            i += 1
            max_t += step_size

        return res

    def neural_sampling(self, c_best, batch_size):
        # normal informed sampling
        samples = self.informed_sample(c_best, batch_size)

        if self.log:
            self.env.utils.visualize_nodes_global(
                self.env_mesh_path,
                self.env_occ_grid,
                samples,
                start_pos=list(self.start),
                goal_pos=list(self.goal),
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, f"raw_samples_{self.loop_cnt}.png"),
            )

        # Neural local sample
        sample_pairs = list(itertools.combinations(samples, 2))[: self.max_neural_samples]
        local_vs = []
        local_gs = []
        local_occ_grids = []
        cur_vs = []
        for sample_pair in sample_pairs:
            s1 = sample_pair[0]
            s2 = sample_pair[1]

            local_g = self.env.utils.global_to_local(s2, s1)
            local_v = self.env.utils.global_to_local(s1, s1)
            local_occ_grid = self.env.get_local_occ_grid(s1)
            local_occ_grid = self.env.utils.add_pos_channels_np(local_occ_grid)

            local_vs.append(local_v)
            local_gs.append(local_g)
            local_occ_grids.append(local_occ_grid)
            cur_vs.append(s1)

        # - sample in a batch
        print(f"Neural sampling: num of samples: {len(samples)}, num of wp: {len(local_vs)}")
        if len(local_vs) > 0:
            local_sampled_wps = self.sampler.sample_batch(
                np.array(local_vs), np.array(local_gs), np.array(local_occ_grids)
            )
            sampled_wps = [
                tuple(self.env.utils.local_to_global(x, cur_v)) for x, cur_v in zip(local_sampled_wps, cur_vs)
            ]

            num_free_wp = 0
            for sample_wp in sampled_wps:
                if self.is_point_free(sample_wp):
                    samples.append(sample_wp)
                    num_free_wp += 1
            print(f"num of free wp: {num_free_wp}")

        if self.log:
            self.env.utils.visualize_nodes_global(
                self.env_mesh_path,
                self.env_occ_grid,
                samples,
                start_pos=list(self.start),
                goal_pos=list(self.goal),
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, f"neural_samples_{self.loop_cnt}.png"),
            )

        # Return informed sampled + neural local samples
        return samples

    def plan(self, pathLengthLimit=INF, refine_time_budget=None):
        # collision_checks = self.env.collision_check_count
        # if time_budget is None:
        #     time_budget = INF
        # if refine_time_budget is None:
        #     refine_time_budget = 10

        while True:
            loop_start_time = time.time()
            if self.num_samples >= self.num_sample_max:
                print("[BIT*]: exit because maximum number of sample reached")
                break

            if time.time() >= self.allowed_time:
                print("[BIT*]: exit because maximum time reached")
                break

            if self.loop_cnt >= self.max_edge_eval:
                print("[BIT*]: exit because maximum number of edge evaluation reached")
                break

            # # Handle the special case where start is directly connected to goal. So no more samples can be drawn
            # if self.g_scores[self.goal] < self.c_min + 1e-4:
            #     end_time_1 = time.time()
            #     self.avg_loop_time += end_time_1 - loop_start_time
            #     self.loop_cnt += 1
            #     continue

            # sampling
            start_time_1 = time.time()
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                self.prune(c_best)
                self.samples.extend(self.neural_sampling(c_best, self.batch_size))
                self.num_samples += self.batch_size
                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)  # change to op priority queue
                q = len(self.vertices) + len(self.samples)
                self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))

            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    _, point = heapq.heappop(self.vertex_queue)
                    self.expand_vertex(point)
                    if len(self.edge_queue) == 0:
                        break
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            if len(self.edge_queue) == 0:
                end_time_1 = time.time()
                self.total_running_time += end_time_1 - loop_start_time
                self.loop_cnt += 1

                end_time_1 = time.time()
                time_taken = end_time_1 - start_time_1
                self.total_vertex_selection_time += time_taken

                continue

            # print("evaluating edges", len(self.vertices), len(self.edges), len(self.edge_queue), self.g_scores[self.goal])
            best_edge_value, best_edge = heapq.heappop(self.edge_queue)

            end_time_1 = time.time()
            time_taken = end_time_1 - start_time_1
            self.total_vertex_selection_time += time_taken

            # vertex extension
            start_time_1 = time.time()
            # Check if this can improve the current solution
            if best_edge_value < self.g_scores[self.goal]:
                start_time_1 = time.time()
                actual_cost_of_edge = self.actual_edge_cost(best_edge[0], best_edge[1])
                end_time_1 = time.time()
                self.col_check_time += end_time_1 - start_time_1
                self.num_col_check += 1

                actual_f_edge = (
                    self.heuristic_cost(self.start, best_edge[0])
                    + actual_cost_of_edge
                    + self.heuristic_cost(best_edge[1], self.goal)
                )
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(best_edge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(best_edge[1]):
                        self.g_scores[best_edge[1]] = actual_g_score_of_point
                        self.edges[best_edge[1]] = best_edge[0]
                        if best_edge[1] not in self.vertices:
                            self.samples.remove(best_edge[1])
                            self.vertices.append(best_edge[1])

                            heapq.heappush(self.vertex_queue, (self.get_point_value(best_edge[1]), best_edge[1]))

                        # Add to networkx graph
                        self.graph.add_edge(best_edge[0], best_edge[1], weight=actual_cost_of_edge)

                        self.edge_queue = [
                            item
                            for item in self.edge_queue
                            if item[1][1] != best_edge[1]
                            or self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], item[1][1])
                            < self.get_g_score(item[1][1])
                        ]
                        heapq.heapify(self.edge_queue)  # Rebuild the priority queue because it will be destroyed after the element is removed
            else:
                self.vertex_queue = []
                self.edge_queue = []

            # print("after adding edges", len(self.vertices), len(self.edges), len(self.edge_queue), self.g_scores[self.goal])

            # if self.g_scores[self.goal] < pathLengthLimit and (time() - init_time > refine_time_budget):
            #     break

            end_time_1 = time.time()
            time_taken = end_time_1 - start_time_1
            self.total_vertex_extension_time += time_taken

            end_time_1 = time.time()
            self.total_running_time += end_time_1 - loop_start_time
            self.loop_cnt += 1

        success = self.g_scores[self.goal] < INF
        path = []
        if success:
            path = nx.shortest_path(self.graph, source=self.start, target=self.goal)
            path = [list(p) for p in path]

        return path

        # return self.samples, self.edges, 0, success, self.T, time.time() - init_time


class NRPBITStarV2(NRPBITStar):
    def __init__(
        self,
        env,
        model_path,
        dim=11,
        occ_grid_dim=[40, 40, 20],
        device=torch.device("cuda"),
        batch_size=100,
        sampling=None,
        log=False,
    ):
        super().__init__(env, model_path, dim, occ_grid_dim, device, batch_size, sampling=sampling, log=log)

        self.sample_pairs = []
        self.neural_sample_bias = 0.8

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            self.sample_pairs.append([point1, point2])
            self.num_edge_col += 1
            return INF
        return self.distance(point1, point2)

    def neural_sampling(self, c_best, batch_size):
        num_nueral_samples = min(int(batch_size * self.neural_sample_bias), len(self.sample_pairs))
        num_informed_samples = batch_size - num_nueral_samples
        print(f"Neural sampling: num informed samples: {num_informed_samples}, num neural samples: {num_nueral_samples}")

        # normal informed sampling
        samples = self.informed_sample(c_best, num_informed_samples)

        if self.log:
            self.env.utils.visualize_nodes_global(
                self.env_mesh_path,
                self.env_occ_grid,
                samples,
                start_pos=list(self.start),
                goal_pos=list(self.goal),
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, f"raw_samples_{self.loop_cnt}.png"),
            )

        # Neural local sample
        local_vs = []
        local_gs = []
        local_occ_grids = []
        cur_vs = []
        for sample_pair in self.sample_pairs[:num_nueral_samples]:
            s1 = sample_pair[0]
            s2 = sample_pair[1]

            local_g = self.env.utils.global_to_local(s2, s1)
            local_v = self.env.utils.global_to_local(s1, s1)
            local_occ_grid = self.env.get_local_occ_grid(s1)
            local_occ_grid = self.env.utils.add_pos_channels_np(local_occ_grid)

            local_vs.append(local_v)
            local_gs.append(local_g)
            local_occ_grids.append(local_occ_grid)
            cur_vs.append(s1)

        # - sample in a batch
        num_free_wp = 0
        if len(local_vs) > 0:
            # start_time = time.time()
            local_sampled_wps = self.sampler.sample_batch(
                np.array(local_vs), np.array(local_gs), np.array(local_occ_grids)
            )
            # end_time = time.time()
            # print(end_time - start_time)

            sampled_wps = [
                tuple(self.env.utils.local_to_global(x, cur_v)) for x, cur_v in zip(local_sampled_wps, cur_vs)
            ]

            for sample_wp in sampled_wps:
                if self.is_point_free(sample_wp):
                    samples.append(sample_wp)
                    num_free_wp += 1

        print(f"Neural sampling: free informed samples: {len(samples)-num_free_wp}, free neural samples: {num_free_wp}")

        if self.log:
            self.env.utils.visualize_nodes_global(
                self.env_mesh_path,
                self.env_occ_grid,
                samples,
                start_pos=list(self.start),
                goal_pos=list(self.goal),
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, f"neural_samples_{self.loop_cnt}.png"),
            )

        self.sample_pairs.clear()
        # Return informed sampled + neural local samples
        return samples

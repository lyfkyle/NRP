# Adopted from https://github.com/rainorangelemon/gnn-motion-planning
import os.path as osp
import numpy as np
import math
import json
import random
import time
import networkx as nx
from sklearn.neighbors import NearestNeighbors

INF = float("inf")

class InformedRRTStar:
    def __init__(self, col_checker_fn, dist_fn, global_sampler, local_sampler, interpolate_fn=None):
        self.global_sampler = global_sampler
        self.local_sampler = local_sampler
        self.dist_fn = dist_fn
        self.col_checker_fn = col_checker_fn
        self.interpolate_fn = interpolate_fn

        self.log_dir = None
        self.n_collision_points = 0
        self.n_free_points = 2
        self.graph = nx.DiGraph()

        self.return_on_path_find = False
        self.add_intermediate_state = False
        self.goal_bias = 0.1

        self.clear()

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
        self.C = np.dot(np.dot(U, np.diag([1] * (self.dimension - 1) + [np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def informed_sample(self, c_best, sample_num, vertices):
        # print("sampling")
        # The following are exactly from informed RRT* paper
        if c_best < float('inf'):
            c_b = math.sqrt(c_best ** 2 - self.c_min ** 2) / 2.0
            r = [c_best / 2.0] + [c_b] * (self.dimension - 1)
            L = np.diag(r)
            x_ball = self.sample_unit_ball()
            random_point = tuple(np.dot(np.dot(self.C, L), x_ball) + self.center_point)
        else:
            random_point = self.get_random_point()

        # self.is_point_free(random_point)

        return random_point

    def is_edge_free(self, v1, v2):
        key1 = (v1, v2)
        key2 = (v2, v1)
        if key1 in self.col_cache:
            return self.col_cache[key1]
        if key2 in self.col_cache:
            return self.col_cache[key2]
        else:
            result = self.col_checker_fn(v1, v2)
            self.col_cache[key1] = result
            self.col_cache[key2] = result

        return result

    def get_random_point(self):
        # point = self.bounds[:, 0] + np.random.random(self.dimension) * self.ranges
        # return tuple(point)
        return tuple(self.global_sampler(None, 1)[0])

    def get_nearest_neighbour(self, nodes, target):
        '''
        return the closest neighbours of v in V
        @param V, a list of vertices, must be a 2D numpy array
        @param v, the target vertice, must be a 2D numpy array
        @return, list, the nearest neighbours
        '''

        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=self.dist_fn).fit(V)
        # distances, indices = nbrs.kneighbors(v)
        # # print("indices {}".format(indices))
        # # return np.take(np.array(V), indices.ravel(), axis=0)[0].tolist()
        # return indices.ravel()[0]
        dlist = self.dist_fn(nodes, target).reshape(-1)
        min_idx = np.argmin(dlist)
        return min_idx

    def clear(self):
        self.env = None
        self.start = None
        self.goal = None

        # This is the tree
        self.graph.clear()
        self.g_scores = dict()

        self.max_extension_num = float('inf')
        self.allowed_time = float('inf')

        self.r = INF
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        self.col_check_time = 0
        self.num_edge_col = 0
        self.total_running_time = 0
        self.total_vertex_selection_time = 0
        self.total_vertex_selection_success_time = 0
        self.total_vertex_selection_fail_time = 0
        self.total_vertex_extension_time = 0
        self.total_vertex_extension_success_time = 0
        self.total_vertex_extension_fail_time = 0
        self.total_succcesful_vertex_extension = 0
        self.num_expansions = 0
        self.rewire_cnt = 0
        self.col_cache = {}

    def setup_planning(self):
        bounds = self.env.robot.get_joint_bounds()
        self.dimension = self.env.robot.num_dim
        self.bounds = np.array(bounds).reshape(-1, 2)
        # print(self.bounds)
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        # print(self.ranges)

        # the parameters for informed sampling
        self.c_min = self.dist_fn(self.start, self.goal)
        print(self.c_min)
        self.center_point = None
        self.C = None

        # add goal to the samples
        self.g_scores[self.goal] = INF
        # self.graph.add_node(self.goal)

        # add start to the tree
        self.g_scores[self.start] = 0
        self.graph.add_node(self.start)

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

    def solve(self, start, goal, allowed_time=float('inf'), max_samples=float('inf')):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.max_extension_num = max_samples
        self.setup_planning()

        if self.log_dir is not None:
            nx.write_graphml(self.graph, osp.join(self.log_dir, "tree_nodes_0.graphml"))

        self.allowed_time = time.time() + allowed_time
        path = self.continue_solve()

        return path

    def get_neighbours(self, sample, K = 5):
        graph_nodes = np.array(list(self.graph.nodes()))
        dlist = self.dist_fn(graph_nodes, sample).reshape(-1)
        dist_cache = {}

        if len(graph_nodes) <= K:
            neighbours = graph_nodes.tolist()
            neighbour_dist = dlist
        else:
            # use radius:
            # neighbours = graph_nodes[dlist < radius].tolist()
            # use k-nearest neighbour
            topK = np.argpartition(dlist, K)[:K]
            neighbours = graph_nodes[topK].tolist()
            neighbour_dist = dlist[topK]

        neighbours = [tuple(n) for n in neighbours]
        for i, n in enumerate(neighbours):
            dist_cache[(n, sample)] = neighbour_dist[i]
            dist_cache[(sample, n)] = neighbour_dist[i]

        return neighbours, dist_cache

    # def get_neighbours(self, sample, K = 5):
    #     graph_nodes = np.array(list(self.graph.nodes()))
    #     if len(graph_nodes) <= K:
    #         neighbours = graph_nodes.tolist()
    #     else:
    #         neigh = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    #         neigh.fit(graph_nodes)
    #         idx = neigh.kneighbors(np.array([sample]), return_distance=False).reshape(-1)
    #         neighbours = graph_nodes[idx]
    #     neighbours = [tuple(n) for n in neighbours]
    #     return neighbours

    def continue_solve(self):
        while True:
            end_time = time.time()
            if end_time > self.allowed_time:
                print("RRT* exit because time limit reached")
                break

            # if self.graph.number_of_nodes() > self.max_num_samples:
            #     break

            if self.num_expansions >= self.max_extension_num:
                print("RRT* exit because number of extension limit reached")
                break

            c_best = self.g_scores[self.goal]
            if c_best < self.c_min:
                self.g_scores[self.goal] = self.c_min
                c_best = self.c_min

            # vertex selection
            start_time_1 = time.time()
            if random.uniform(0, 1) < self.goal_bias and not self.graph.has_node(self.goal):
                global_sample = self.goal
            else:
                global_sample = self.informed_sample(c_best, 1, None)

            graph_nodes = np.array(list(self.graph.nodes()))
            # graph_nodes_base = graph_nodes[:, :3]
            # g_s_base = np.array(g_s)[:, :3]
            # print(graph_nodes_base, g_s_base)
            # v = self.get_nearest_neighbour(np.array(list(self.graph.nodes())), np.array(g_s))
            indice = self.get_nearest_neighbour(graph_nodes, np.array(global_sample))
            v = graph_nodes[indice].tolist()
            # print(v)

            end_time_1 = time.time()
            vs_time_taken = end_time_1 - start_time_1
            self.total_vertex_selection_time += vs_time_taken
            # print("vertex selection takes: {}".format(time_taken))

            # vertex expansion
            start_time_1 = time.time()
            paths, _, _ = self.local_sampler(v, global_sample)
            end_time_1 = time.time()
            ve_time_taken = end_time_1 - start_time_1
            self.total_vertex_extension_time += ve_time_taken
            if len(paths[0]) <= 1:
                self.total_vertex_extension_fail_time += ve_time_taken
                self.total_vertex_selection_fail_time += vs_time_taken
            else:
                self.total_succcesful_vertex_extension += 1
                self.total_vertex_extension_success_time += ve_time_taken
                self.total_vertex_selection_success_time += vs_time_taken
            # print("vertex expansion takes: {}".format(ve_time_taken))

            # start_time_1 = time.time()
            for path in paths:
                if len(path) <= 1:
                    # print("RRT:No extension happens!!!")
                    continue

                path = [tuple(p) for p in path]
                # print(len(path))
                # assert len(path) <= 3
                x_min = tuple(v)
                for sample in path[1:]:
                    # Shortcut: Connect to best neighbour
                    c_min = self.g_scores[x_min] + self.dist_fn(x_min, sample).item()
                    # start_time_1 = time.time()
                    x_nears, dist_cache = self.get_neighbours(sample)
                    # end_time_1 = time.time()
                    # time_taken = end_time_1 - start_time_1
                    # print("get_neighbour takes: {}".format(time_taken), len(x_nears))

                    # # print(x_nears)
                    for x_near in x_nears:
                        c_x = self.g_scores[x_near] + dist_cache[(x_near, sample)]
                        if c_x < c_min:
                            if self.col_checker_fn(x_near, sample):
                                x_min = x_near
                                c_min = c_x

                    # Add edge
                    if not self.add_intermediate_state:
                        self.graph.add_node(sample)
                        self.graph.add_edge(x_min, sample)
                        self.g_scores[sample] = c_min
                    else:
                        path_interpolated, step_size = self.interpolate_fn(x_min, sample)
                        for i in range(1, len(path_interpolated)):
                            prev_p = tuple(path_interpolated[i - 1])
                            cur_p = tuple(path_interpolated[i])
                            self.graph.add_node(cur_p)
                            self.graph.add_edge(prev_p, cur_p)
                            self.g_scores[cur_p] = c_min + step_size * i
                    # end_time_1 = time.time()
                    # time_taken = end_time_1 - start_time_1
                    # print("shortcut takes: {}".format(time_taken))

                    # Rewire
                    for x_near in x_nears:
                        c_near = self.g_scores[x_near]
                        c_new = self.g_scores[sample] + dist_cache[(sample, x_near)]
                        if c_new < c_near:
                            if self.col_checker_fn(sample, x_near):
                                x_parent = list(self.graph.predecessors(x_near))[0]
                                self.graph.remove_edge(x_parent, x_near)
                                self.graph.add_edge(sample, x_near)
                                self.rewire_cnt += 1

                    # end_time_1 = time.time()
                    # time_taken = end_time_1 - start_time_1
                    # print("rewires takes: {}".format(time_taken))

                    x_min = sample

            end_time_1 = time.time()
            time_taken = end_time_1 - end_time
            # print("tree update takes: {}".format(time_taken))
            self.total_running_time += time_taken
            self.num_expansions += 1

            if self.log_dir is not None:
                # tree_nodes = list(self.graph.nodes())
                # with open(osp.join(self.log_dir, "tree_nodes_{}.json".format(self.num_sampler_called)), 'w') as f:
                #     json.dump(tree_nodes, f)
                print(len(self.graph.nodes()))
                nx.write_graphml(self.graph, osp.join(self.log_dir, "tree_nodes_{}.graphml".format(self.num_expansions)))

            # return if solution found
            if self.graph.has_node(self.goal) and self.return_on_path_find:
                break

            # end_time_1 = time.time()
            # time_taken = end_time_1 - end_time
            # print("tree update takes: {}".format(time_taken))

        if self.graph.has_node(self.goal):
            p = nx.shortest_path(self.graph, source=self.start, target=self.goal)
            return p
        else:
            return []
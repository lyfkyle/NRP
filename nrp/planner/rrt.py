import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import time
import random
import math
import json

class RRT(object):
    def __init__(self, col_checker, dist_fn, global_sampler, local_sampler):
        self.global_sampler = global_sampler
        self.local_sampler = local_sampler
        self.dist_fn = dist_fn
        self.col_checker = col_checker
        self.graph = nx.Graph()
        self.sample_cnt = 0
        self.max_extension_num = float('inf')
        self.num_expansions = 0

        self.return_on_path_find = True
        self.log_dir = None
        self.goal_bias = 0.2

    def set_sampler(self, sampler):
        self.local_sampler = sampler

    def set_occ_grid(self, occ_grid):
        self.occ_grid = occ_grid

    def set_max_num_samples(self, num):
        self.max_extension_num = num

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
        self.start = None
        self.goal = None
        self.graph.clear()
        self.total_running_time = 0
        self.num_expansions = 0
        self.start_time = None
        self.allowed_time = None
        self.max_extension_num = None

    def solve(self, start, goal, allowed_time = 1.0, max_extensions=float('inf')):
        self.clear()
        self.start = tuple(start)
        self.goal = tuple(goal)
        # self.graph.clear()
        self.total_running_time = 0
        self.total_vertex_selection_time = 0
        self.total_vertex_selection_success_time = 0
        self.total_vertex_selection_fail_time = 0
        self.total_vertex_extension_time = 0
        self.total_vertex_extension_success_time = 0
        self.total_vertex_extension_fail_time = 0
        self.total_succcesful_vertex_extension = 0
        self.num_expansions = 0
        self.max_extension_num = max_extensions
        self.graph.add_node(self.start)
        # self.graph.add_node("n{}".format(self.graph.number_of_nodes()), coords=','.join(map(str, self.start)))

        if self.log_dir is not None:
            # tree_nodes = list(self.graph.nodes())
            # with open(osp.join(self.log_dir, "tree_nodes_0.json"), 'w') as f:
            #     json.dump(tree_nodes, f)

            nx.write_graphml(self.graph, osp.join(self.log_dir, "tree_nodes_0.graphml"))

        self.allowed_time = time.time() + allowed_time
        return self.continue_solve()

    def continue_solve(self):
        goal = self.goal
        while True:
            end_time = time.time()
            if end_time > self.allowed_time:
                print("RRT exit because time limit reached")
                break

            # if self.graph.number_of_nodes() > self.max_num_samples:
            #     break

            if self.num_expansions >= self.max_extension_num:
                print("RRT exit because number of extension limit reached")
                break

            # vertex selection
            start_time_1 = time.time()
            if random.uniform(0, 1) < self.goal_bias and not self.graph.has_node(self.goal):
                g_s = [goal]
            else:
                g_s = self.global_sampler(None, 1)

            graph_nodes = np.array(list(self.graph.nodes()))
            # graph_nodes_base = graph_nodes[:, :3]
            # g_s_base = np.array(g_s)[:, :3]
            # print(graph_nodes_base, g_s_base)
            # v = self.get_nearest_neighbour(np.array(list(self.graph.nodes())), np.array(g_s))
            indice = self.get_nearest_neighbour(graph_nodes, np.array(g_s))
            v = graph_nodes[indice].tolist()
            # print(v)

            end_time_1 = time.time()
            vs_time_taken = end_time_1 - start_time_1
            self.total_vertex_selection_time += vs_time_taken
            # print("vertex selection takes: {}".format(time_taken))

            # vertex expansion
            start_time_1 = time.time()
            paths, _, _ = self.local_sampler(v, g_s[0])
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
            # print("extension takes: {}".format(time_taken))

            # start_time_1 = time.time()
            for path in paths:
                if len(path) <= 1:
                    # print("RRT:No extension happens!!!")
                    continue

                prev_sample = v
                for sample in path[1:]:
                    # utils.visualize_nodes(self.occ_grid, [sample], None, None, v, g_s[0])
                    self.graph.add_node(tuple(sample))
                    self.graph.add_edge(tuple(prev_sample), tuple(sample))
                    prev_sample = sample

                # if self.col_checker(prev_sample, goal) < float('inf'):
                #     self.graph.add_node(tuple(goal))
                #     self.graph.add_edge(tuple(prev_sample), tuple(goal))

            end_time_1 = time.time()
            time_taken = end_time_1 - end_time
            # print("[RRT]: one loop takes {}".format(time_taken))
            self.total_running_time += time_taken
            self.num_expansions += 1

            if self.log_dir is not None:
                # tree_nodes = list(self.graph.nodes())
                # with open(osp.join(self.log_dir, "tree_nodes_{}.json".format(self.num_sampler_called)), 'w') as f:
                #     json.dump(tree_nodes, f)
                nx.write_graphml(self.graph, osp.join(self.log_dir, "tree_nodes_{}.graphml".format(self.num_expansions)))

            # return if solution found
            if self.graph.has_node(self.goal) and self.return_on_path_find:
                # p = nx.shortest_path(self.graph, source=self.start, target=self.goal)
                # return p
                break

            # end_time_1 = time.time()
            # time_taken = end_time_1 - start_time_1
            # print("tree update takes: {}".format(time_taken))

        if self.graph.has_node(self.goal):
            p = nx.shortest_path(self.graph, source=self.start, target=self.goal)
            return p
        else:
            return []

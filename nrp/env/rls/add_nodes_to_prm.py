import os.path as osp
import networkx as nx
import numpy as np
import math

from nrp.env.rls.rls_env import RLSEnv
from nrp.env.rls import utils
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))

PRM_CONNECT_RADIUS = 2.0


def state_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def process_env():
    env = RLSEnv(gui=False)

    env_dir = osp.join(ROOT_DIR, "env/rls/map")
    print("Process: generating env:{}".format(env_dir))

    # env
    occ_grid = utils.get_occ_grid(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)

    # utils.visualize_nodes_global(osp.join(env_dir, "env_final.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))

    # states
    low = env.robot.get_joint_lower_bounds()
    high = env.robot.get_joint_higher_bounds()

    print(low, high)

    G = nx.read_graphml(osp.join(env_dir, "dense_g_orig.graphml"))

    i = G.number_of_nodes()

    assert env.pb_ompl_interface.is_state_valid(node_poss[2])
    assert utils.is_edge_free(env, node_poss[2], node_poss[8])
    assert utils.is_edge_free(env, node_poss[8], node_poss[9])
    assert utils.is_edge_free(env, node_poss[9], node_poss[3])

    print(i)
    for node_pos in node_poss:
        free_nodes = [n for n in G.nodes() if not G.nodes[n]["col"]]
        node = "n{}".format(i)
        G.add_node(node, coords=",".join(map(str, node_pos)), col=False)

        for f_node in free_nodes:
            s1 = state_to_numpy(G.nodes[f_node]["coords"])
            s2 = node_pos

            if math.fabs(s2[0] - s1[0]) > PRM_CONNECT_RADIUS or math.fabs(s2[1] - s1[1]) > PRM_CONNECT_RADIUS:
                continue

            if utils.is_edge_free(env, s1, s2):
                G.add_edge(node, f_node, weight=utils.calc_edge_len(s1, s2))

        i += 1

    print(i)
    nx.write_graphml(G, osp.join(env_dir, "dense_g.graphml"))


if __name__ == "__main__":
    node_poss = [
        [
            1.0905469025325267,
            4,
            -1.024216337017743,
            0.2,
            1.3194016147644043,
            1.4002991243103027,
            -0.20051024465408326,
            1.519276445135498,
            -0.0002711178322315211,
            1.6581695629821778,
            0.00046484643704652797,
        ],
        [
            5.2,
            1.05,
            -1.7043219922841046,
            0.184,
            -0.8968170951812744,
            0.38978915650634766,
            -1.3763065746292114,
            -2.1162578413635256,
            -3.0271986299713136,
            -1.2057725833190918,
            -2.035894721689453,
        ],
        [8.4, 3.59, 1.0, 0.39, 0.202, -0.681, -0.676, 0.928, 0.543, 1.503, 0.0],
        [
            8.50,
            6.15,
            1.628,
            0.366,
            -1.5460745166748047,
            -0.7265653089782715,
            -1.097505491064453,
            -1.3891512701660156,
            1.3726417249481202,
            0.39647036330566404,
            0.8065717518899536,
        ],
        [
            11.3,
            2.1,
            -3.0,
            0.31,
            0.3974791992694855,
            -0.320060677935791,
            -0.3124906530841827,
            0.3534510781616211,
            0.31266097486419675,
            1.5672810627685547,
            2.361261040029297,
        ],
        [
            3.7,
            2.2,
            1.0239268783749709,
            0.12,
            -0.4822588335483551,
            -0.44277853529663086,
            -2.2594961336120605,
            -1.161355192437744,
            -2.381776409741211,
            0,
            1.7729792893502807,
        ],
        [
            7.885694229602814,
            3.207532024383545,
            -2.2585325241088867,
            0.22938035428524017,
            0.26921966671943665,
            -0.1508634388446808,
            -0.8172637820243835,
            0.7204707264900208,
            0.30389073491096497,
            0.3844086229801178,
            -0.45346903800964355,
        ],
        [
            7.329188703175873,
            3.360751140676661,
            2.7499141631487687,
            0.2089652344584465,
            0.4184368997812271,
            -0.3789847642183304,
            -0.028840333223342896,
            0.2634991705417633,
            -0.48648689687252045,
            -0.5385686010122299,
            -0.6312605738639832,
        ],
        [
            7.39244773387909,
            4.540188944339752,
            0.2908107340335846,
            0.21234756708145142,
            -0.6886276006698608,
            -0.6987711191177368,
            -2.5937294960021973,
            -0.45270201563835144,
            2.5435028076171875,
            -0.3082909882068634,
            -1.8911036252975464,
        ],
        [
            7.383994853779023,
            4.604865978477392,
            1.71095969201914,
            0.21117408502817195,
            -0.700543953637536,
            -0.6617985666316235,
            -2.539171118893045,
            -0.44405607127626834,
            2.348687213711455,
            -0.26380570881505966,
            -1.7472795119878488,
        ],
    ]
    process_env()

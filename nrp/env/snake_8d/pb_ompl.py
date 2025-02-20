#!/usr/bin/env python

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(abspath(__file__)), "../../third_party/ompl/py-bindings"))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

import pybullet as p
import time
from itertools import product

from . import pb_utils

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 2.0
MAXIMUM_STEP = 10000


class PbOMPLRobot:
    def __init__(self, id) -> None:
        # Public attributes
        self.id = id
        self.num_dim = p.getNumJoints(id)  # by default, all joints are actuated
        self.joint_idx = list(range(self.num_dim))

        self.joint_bounds = [[-float("inf"), float("inf")]] * self.num_dim
        self.reset()

    def get_joint_bounds(self):
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id)
            if joint_info.jointLowerLimit < joint_info.jointUpperLimit:
                self.joint_bounds[i][0] = joint_info.jointLowerLimit
                self.joint_bounds[i][1] = joint_info.jointUpperLimit
        return self.joint_bounds

    def get_cur_state(self):
        return self.state

    def set_state(self, state):
        """
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        """
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        """
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        """
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        """
        This will be called by the internal OMPL planner
        """
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        """
        Optional, Set custom state sampler.
        """
        self.state_sampler = state_sampler


class PbOMPL:
    def __init__(self, robot, obstacles=[], interpolate_num=INTERPOLATE_NUM) -> None:
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.interpolate_num = interpolate_num
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT")  # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # if not pb_utils.all_between(lower_limits, state, upper_limits):
        # pass
        # print(lower_limits, q, upper_limits)
        # print('Joint limits violated')
        # return True

        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if pb_utils.pairwise_link_collision(
                self.robot_id, link1, self.robot_id, link2
            ):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False
        for body1, body2 in self.check_body_pairs:
            if pb_utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(
        self, robot, obstacles, self_collisions=True, allow_collision_links=[]
    ):
        self.check_link_pairs = (
            pb_utils.get_self_link_pairs(robot.id, robot.joint_idx)
            if self_collisions
            else []
        )
        moving_links = frozenset(
            [
                item
                for item in pb_utils.get_moving_links(robot.id, robot.joint_idx)
                if not item in allow_collision_links
            ]
        )
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))
        # print(self.check_body_pairs)

    def set_planner(self, planner_name):
        if planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTDof":
            self.planner = og.RRTDof(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "BITstarDof":
            self.planner = og.BITstarDof(self.ss.getSpaceInformation())
        else:
            print("{} not recognized".format(planner_name))
            return

        self.ss.setPlanner(self.planner)
        # print(self.planner.params())

    def plan_start_goal(
        self, start, goal, allowed_time=DEFAULT_PLANNING_TIME, interpolate=False
    ):
        # print("start_planning")
        # print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        # iter_term_cond = IterationTerminationCondition(10000)
        # timed_term_cond = ob.timedPlannerTerminationCondition(allowed_time)
        # term_cond = ob.plannerOrTerminationCondition(iter_term_cond, timed_term_cond)

        solved = self.ss.solve(allowed_time, MAXIMUM_STEP)
        res = False
        sol_path_list = None
        if solved.asString() == "Exact solution":
            print("Found exact solution:")
            sol_path_geometric = self.ss.getSolutionPath()
            if interpolate:
                sol_path_geometric.interpolate(self.interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            # for sol_path in sol_path_list:
            #     assert self.is_state_valid(sol_path)
            res = True
        elif solved.asString() == "Approximate solution":
            print("Approximate solution found")
            sol_path_geometric = self.ss.getSolutionPath()
            if interpolate:
                sol_path_geometric.interpolate(self.interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            # for sol_path in sol_path_list:
            #     assert self.is_state_valid(sol_path)
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME, interpolate=False):
        start = self.robot.get_cur_state()
        return self.plan_start_goal(
            start, goal, allowed_time=allowed_time, interpolate=interpolate
        )

    def execute(self, path):
        for q in path:
            self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)

    def clear(self):
        self.ss.clear()

    def visualize_goal(self, goal):
        # TODO
        pass

    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]


# class IterationTerminationCondition(ob.PlannerTerminationCondition):
#     def __init__(self, max_cnt) -> None:
#         super().__init__(self.dummy)
#         self._cnt = 0
#         self._max_cnt = max_cnt

#     def eval(self):
#         self._cnt += 1
#         return self._cnt >= self._max_cnt

#     def dummy(self) -> bool:
#         return True


class PbOMPLPRM:
    def __init__(self, robot, obstacles=[], sim_id=0) -> None:
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.sim_id = sim_id
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        # construct an instance of space information from this state space
        self.si = ob.SpaceInformation(self.space)
        # set state validity checking for this space
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        # create a planner for the defined space
        self.planner = og.PRM(self.si)
        # self.planner = og.PRMstar(self.si)
        pdef = ob.ProblemDefinition(self.si)
        self.planner.setProblemDefinition(pdef)
        self.planner.setup()

        self.set_obstacles(obstacles)

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # if not pb_utils.all_between(lower_limits, state, upper_limits):
        # pass
        # print(lower_limits, q, upper_limits)
        # print('Joint limits violated')
        # return True

        if not isinstance(state, list):
            state = self.state_to_list(state)

        self.robot.set_state(state)
        for link1, link2 in self.check_link_pairs:
            if pb_utils.pairwise_link_collision(
                self.sim_id, self.robot_id, link1, self.robot_id, link2
            ):
                # print("self collision")
                return False
        for body1, body2 in self.check_body_pairs:
            # start_time = time.time()
            res = pb_utils.pairwise_collision(self.sim_id, body1, body2)
            # end_time = time.time()
            # print("col check takes: {}".format(end_time - start_time))
            if res:
                # print("body collision", body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(
        self, robot, obstacles, self_collisions=True, allow_collision_links=[]
    ):
        self.check_link_pairs = (
            pb_utils.get_self_link_pairs(self.sim_id, robot.id, robot.joint_idx)
            if self_collisions
            else []
        )
        moving_links = set(
            [
                item
                for item in pb_utils.get_moving_links(
                    self.sim_id, robot.id, robot.joint_idx
                )
                if not item in allow_collision_links
            ]
        )
        moving_links.add(-1)  # add base_link
        # print(moving_links)
        # moving_bodies = [(robot.id, moving_links)]
        moving_bodies = [robot.id]
        self.check_body_pairs = list(product(moving_bodies, obstacles))
        # print(self.check_body_pairs)

    def construct_prm(self, allowed_time=5.0):
        print("constructing prm")
        timed_term_cond = ob.timedPlannerTerminationCondition(allowed_time)
        self.planner.constructRoadmap(timed_term_cond)

    def get_planner_data(self, file_path):
        planner_data = ob.PlannerData(self.si)
        self.planner.getPlannerData(planner_data)
        graphml = planner_data.printGraphML()
        with open(file_path, "w") as f:
            f.write(graphml)

    def plan_start_goal(self, start, goal, allowed_time=2.0, interpolate=False):
        # print("start_planning")
        # print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # clear previous inputs
        self.planner.clearQuery()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        # create a problem instance
        pdef = ob.ProblemDefinition(self.si)
        # set the start and goal states
        pdef.setStartAndGoalStates(s, g)
        # set the problem we are trying to solve for the planner
        self.planner.setProblemDefinition(pdef)
        # perform setup steps for the planner
        # self.planner.setup()
        # print the settings for this space
        # print(self.si.settings())
        # print the problem settings
        # print(pdef)
        # attempt to solve the problem within one second of planning time
        solved = self.planner.solve(allowed_time)

        # attempt to solve the problem within allowed planning time
        # iter_term_cond = IterationTerminationCondition(10000)
        # timed_term_cond = ob.timedPlannerTerminationCondition(allowed_time)
        # term_cond = ob.plannerOrTerminationCondition(iter_term_cond, timed_term_cond)

        res = False
        sol_path_list = None
        if solved.asString() == "Exact solution":
            print("Found exact solution:")
            sol_path_geometric = pdef.getSolutionPath()
            if interpolate:
                sol_path_geometric.interpolate(self.interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            # for sol_path in sol_path_list:
            #     assert self.is_state_valid(sol_path)
            res = True
        elif solved.asString() == "Approximate solution":
            print("Approximate solution found")
            # sol_path_geometric = pdef.getSolutionPath()
            # sol_path_geometric.interpolate(self.interpolate_num)
            # sol_path_states = sol_path_geometric.getStates()
            # sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            # for sol_path in sol_path_list:
            #     assert self.is_state_valid(sol_path)
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time=2.0, interpolate=False):
        start = self.robot.get_cur_state()
        return self.plan_start_goal(
            start, goal, allowed_time=allowed_time, interpolate=interpolate
        )

    def clear(self):
        self.planner.clear()

    def visualize_goal(self, goal):
        # TODO
        pass

    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]

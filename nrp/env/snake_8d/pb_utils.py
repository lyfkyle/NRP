'''
Adopted from
https://github.com/StanfordVL/iGibson/blob/master/igibson/external/pybullet_tools/utils.py
Original author:
Caelan Reed Garrett. PyBullet Planning. https://pypi.org/project/pybullet-planning/. 2018.
'''

from __future__ import print_function

from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count

BASE_LINK = -1
MAX_DISTANCE = 0.

def pairwise_link_collision(p, body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2)) != 0  # getContactPoints

def pairwise_collision(p, body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(p, body1)
        body2, links2 = expand_links(p, body2)
        return any_link_pair_collision(p, body1, links1, body2, links2, **kwargs)
    return body_collision(p, body1, body2, **kwargs)

def expand_links(p, body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(p, body)
    return body, links

def any_link_pair_collision(p, body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(p, body1)
    if links2 is None:
        links2 = get_all_links(p, body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(p, body1, link1, body2, link2, **kwargs):
            # print('body {} link {} body {} link {}'.format(body1, link1, body2, link2))
            return True
    return False

def body_collision(p, body1, body2, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0  # getContactPoints`

def get_self_link_pairs(p, body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(p, body, joints)
    fixed_links = list(set(get_joints(p, body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(p, body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(p, body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_moving_links(p, body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(p, joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(p, body, link))
    return list(moving_links)

def get_moving_pairs(p, body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(p, body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(p, body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(p, body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2


#####################################

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(p, body, joint):
    return JointInfo(*p.getJointInfo(body, joint))

def child_link_from_joint(p, joint):
    return joint  # link

def get_num_joints(p, body):
    return p.getNumJoints(body)

def get_joints(p, body):
    return list(range(get_num_joints(p, body)))

get_links = get_joints

def get_all_links(p, body):
    return [BASE_LINK] + list(get_links(p, body))

def get_link_parent(p, body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(p, body, link).parentIndex

def get_all_link_parents(p, body):
    return {link: get_link_parent(p, body, link) for link in get_links(p, body)}

def get_all_link_children(p, body):
    children = {}
    for child, parent in get_all_link_parents(p, body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(p, body, link):
    children = get_all_link_children(p, body)
    return children.get(link, [])


def get_link_ancestors(p, body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(p, body, link)
    if parent is None:
        return []
    return get_link_ancestors(p, body, parent) + [parent]


def get_joint_ancestors(p, body, joint):
    link = child_link_from_joint(p, joint)
    return get_link_ancestors(p, body, link) + [link]

def get_link_descendants(p, body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(p, body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(p, body, child, test=test))
    return descendants


def get_link_subtree(p, body, link, **kwargs):
    return [link] + get_link_descendants(p, body, link, **kwargs)

def are_links_adjacent(p, body, link1, link2):
    return (get_link_parent(p, body, link1) == link2) or \
           (get_link_parent(p, body, link2) == link1)



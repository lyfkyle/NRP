import os.path as osp
from lxml import etree

CUR_DIR = osp.dirname(osp.abspath(__file__))

urdf_file_path = osp.join(CUR_DIR, "../robot_model/fetch.urdf")

# read original xacro
tree = etree.parse(urdf_file_path)
root = tree.getroot()

# get link inertial origin
origins = root.xpath('/robot/link/inertial/origin', namespaces=root.nsmap)
for origin in origins:
    xyz = origin.get('xyz').split()
    new_xyz = [str(float(x) * 0.5) for x in xyz]
    new_xyz = " ".join(new_xyz)
    origin.set("xyz", new_xyz)

origins = root.xpath('/robot/link/visual/origin', namespaces=root.nsmap)
for origin in origins:
    xyz = origin.get('xyz').split()
    new_xyz = [str(float(x) * 0.5) for x in xyz]
    new_xyz = " ".join(new_xyz)
    origin.set("xyz", new_xyz)

origins = root.xpath('/robot/link/collision/origin', namespaces=root.nsmap)
for origin in origins:
    xyz = origin.get('xyz').split()
    new_xyz = [str(float(x) * 0.5) for x in xyz]
    new_xyz = " ".join(new_xyz)
    origin.set("xyz", new_xyz)

meshes = root.xpath('/robot/link/visual/geometry/mesh', namespaces=root.nsmap)
for mesh in meshes:
    mesh.set("scale", "0.5 0.5 0.5")

meshes = root.xpath('/robot/link/collision/geometry/mesh', namespaces=root.nsmap)
for mesh in meshes:
    mesh.set("scale", "0.5 0.5 0.5")

origins = root.xpath('/robot/joint/origin', namespaces=root.nsmap)
for origin in origins:
    xyz = origin.get('xyz').split()
    new_xyz = [str(float(x) * 0.5) for x in xyz]
    new_xyz = " ".join(new_xyz)
    origin.set("xyz", new_xyz)

# TORSO lift joint upper TODO

## generate .yaml file
# convert xml to dict
xmlstr = etree.tostring(root, encoding='utf-8', method='xml') # xmltodict won't accept original xml for some reason.
                                                                # This is a workaround

tree.write('output.xml', pretty_print=True)
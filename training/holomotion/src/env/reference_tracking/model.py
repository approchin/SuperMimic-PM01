# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Build a reference G1 MuJoCo model without changing the source robot asset."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .robot import G1_JOINT_NAMES, _kd, _kp, motor_of


def build_model(urdf, output, *, self_collisions=True, robot_spec=None):
    if robot_spec is not None and robot_spec.kind == 'pm01':
        return build_pm01_model(robot_spec, output)
    urdf, output = Path(urdf).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(urdf).getroot()
    extension = robot.find("mujoco")
    if extension is not None:
        robot.remove(extension)
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        balanceinertia="true",
        discardvisual="false",
        fusestatic="false",
        strippath="false",
    )
    for mesh in robot.findall(".//mesh"):
        mesh.set(
            "filename", str((urdf.parent / mesh.get("filename")).resolve())
        )
    compiled = mujoco.MjModel.from_xml_string(
        ET.tostring(robot, encoding="unicode")
    )
    intermediate = output.with_suffix(".compiled.xml")
    mujoco.mj_saveLastXML(str(intermediate), compiled)
    root = ET.parse(intermediate).getroot()
    pelvis = root.find('.//body[@name="pelvis"]')
    if pelvis is None:
        raise ValueError("URDF conversion did not retain the pelvis link")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep="0.005",
        integrator="implicitfast",
        cone="pyramidal",
        iterations="10",
        ls_iterations="10",
    )
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for name in G1_JOINT_NAMES:
        effort, _, armature = motor_of(name)
        joint = root.find(f'.//joint[@name="{name}"]')
        if joint is None:
            raise ValueError(f"URDF conversion lost joint {name}")
        joint.attrib.update(
            armature=str(armature), damping="0", frictionloss="0"
        )
        ET.SubElement(
            actuator,
            "general",
            name=name + "_drive",
            joint=name,
            gainprm=str(_kp(armature)),
            biastype="affine",
            biasprm=f"0 {-_kp(armature)} {-_kd(armature)}",
            forcelimited="true",
            forcerange=f"{-effort} {effort}",
        )
    world = root.find("worldbody")
    # Explicit masks separate disabling the penalty from disabling collisions.
    for geom in root.findall(".//body//geom"):
        if (
            geom.get("contype", "1") == "0"
            and geom.get("conaffinity", "1") == "0"
        ):
            continue
        geom.attrib.update(friction="0.8 0.005 0.0001")
        if not self_collisions:
            geom.attrib.update(contype="1", conaffinity="0")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="0.3 0.005 0.0001",
        contype="0" if not self_collisions else "1",
        conaffinity="1",
    )
    ET.indent(root)
    output.write_text(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(output))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(
            f"Unexpected G1 model dimensions: {model.nq}/{model.nv}/{model.nu}"
        )
    if [model.joint(i).name for i in range(1, model.njnt)] != G1_JOINT_NAMES:
        raise ValueError(
            "MuJoCo joint order differs from the declared reference data order"
        )
    return model


def build_pm01_model(spec, output):
    """Owned model copy: original SDK geometry/inertia with source PM01 PDs.

    MuJoCo torque saturation/position drives are not a claim of PhysX parity;
    in particular PhysX motor velocity limits have no identical native cap.
    """
    output = Path(output).resolve();output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(spec.mjcf).resolve()
    original = mujoco.MjModel.from_xml_path(str(source))
    intermediate = output.with_suffix('.source_compiled.xml')
    mujoco.mj_saveLastXML(str(intermediate), original)
    root = ET.parse(intermediate).getroot()
    compiler = root.find('compiler')
    meshdir = Path(compiler.get('meshdir', ''))
    if not meshdir.is_absolute(): meshdir = source.parent/meshdir
    for mesh in root.findall('.//mesh'):
        path = Path(mesh.get('file'))
        mesh.set('file', str(path if path.is_absolute() else (meshdir/path).resolve()))
    texturedir=Path(compiler.get('texturedir',''))
    if not texturedir.is_absolute():texturedir=source.parent/texturedir
    for texture in root.findall('.//texture'):
        if texture.get('file'):
            path=Path(texture.get('file'))
            texture.set('file',str(path if path.is_absolute() else (texturedir/path).resolve()))
    compiler.attrib.pop('texturedir',None)
    compiler.attrib.pop('meshdir', None)
    root.find('option').set('timestep', '0.005')
    old = root.find('actuator')
    if old is not None: root.remove(old)
    actuator = ET.SubElement(root, 'actuator')
    for i, name in enumerate(spec.joint_names):
        joint = root.find(f'.//joint[@name="{name}"]')
        if joint is None: raise ValueError('Original PM01 joint missing: '+name)
        joint.attrib.update(armature=str(spec.armature[i]), damping='0', frictionloss='0')
        ET.SubElement(actuator, 'general', name=name+'_drive', joint=name,
            gainprm=str(spec.kp[i]), biastype='affine', biasprm=f'0 {-spec.kp[i]} {-spec.kd[i]}',
            forcelimited='true', forcerange=f'{-spec.effort[i]} {spec.effort[i]}')
    if root.find('.//geom[@name="floor"]') is None:
        ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane', size='0 0 .1', friction='1 .005 .0001')
    # The asset keyframe refers to original torque actuators; retain pose data
    # only and let each physical reset write its actual PD target explicitly.
    keyframe = root.find('keyframe')
    if keyframe is not None: root.remove(keyframe)
    ET.indent(root);output.write_text(ET.tostring(root, encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(output))
    actual = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if (model.nq,model.nv,model.nu) != (spec.nq,spec.nv,spec.applied_actions) or actual != list(spec.joint_names):
        raise ValueError('Native PM01 model/named-joint dimensions differ')
    return model

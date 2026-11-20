#!/usr/bin/env python3
"""Render a recorded PM01 control sequence and its synchronized reference side by side.

Supply a compatible visual MJCF with its assets and a numeric NPZ as specified in
docs/videos.md. No simulator stepping or policy inference takes place during replay.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from supermimic_pm01.policy_core import JOINT_NAMES

SIZE = (1280, 720)
PANE = (616, 452)
PANEL_Y = 119
DARK = '#121e29'
TEXT = '#f0f4f3'
MUTED = '#b2c0c7'
ACTUAL = '#65e0c2'
REFERENCE = '#ffbc72'
GRID = '#334553'
ACTOR_SHA256 = '56436a9977e47279074ea620b0da806df179af709b5475f82cecd13942c95395'


@dataclass
class Trace:
    qpos: np.ndarray
    ref_qpos: np.ndarray
    planned_steps: int
    completed: bool
    control_hz: int
    physics_hz: int
    stop_reason: str

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['schema'].item()) != 'pm01-public-render-input-v1':
                raise ValueError('Unexpected trace schema')
            if str(data['onnx_sha256'].item()) != ACTOR_SHA256:
                raise ValueError('The trace identifies a different actor')
            if tuple(data['joint_names'].tolist()) != JOINT_NAMES:
                raise ValueError('Trace joints must use native PM01 order')
            tick_rate = int(data['control_hz'].item())
            physics_rate = int(data['physics_hz'].item())
            plan = int(data['planned_steps'].item())
            qpos, ref = data['qpos'], data['ref_qpos']
            steps = len(qpos)-1
            if (tick_rate!=50 or physics_rate!=200 or plan<1 or not 1<=steps<=plan
                    or qpos.shape!=(steps+1,31) or ref.shape!=qpos.shape
                    or not np.isfinite(qpos).all() or not np.isfinite(ref).all()):
                raise ValueError('Expected two finite PM01 pose streams at 50 Hz')
            if (not np.array_equal(data['step'], np.arange(steps+1))
                    or not np.allclose(data['time_s'], np.arange(steps+1)/tick_rate, rtol=0, atol=1e-8)):
                raise ValueError('A control sample is missing, repeated, or mistimed')
            reason = str(data['stop_reason'].item())
            complete = bool(data['completed'].item())
            if complete != (reason=='reference_end') or (complete and steps!=plan):
                raise ValueError('Outcome disagrees with the trace length or stop reason')
            for poses in (qpos, ref):
                if not np.allclose(np.linalg.norm(poses[:,3:7], axis=1),1.0,rtol=0,atol=2e-4):
                    raise ValueError('Invalid free-root quaternion')
            return cls(qpos,ref,plan,complete,tick_rate,physics_rate,reason)


def load_visual_model(path):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.visual.global_.offwidth = max(PANE[0], spec.visual.global_.offwidth)
    spec.visual.global_.offheight = max(PANE[1], spec.visual.global_.offheight)
    spec.visual.quality.offsamples = 2
    # Adjust the studio background and the fixed floor grid. Robot meshes and
    # kinematic poses are unchanged. These settings are used only for rendering.
    floor_texture = next((t for t in spec.textures if t.name=='groundplane'), None)
    if floor_texture is not None:
        floor_texture.rgb1 = [.51,.57,.59]
        floor_texture.rgb2 = [.58,.63,.64]
        floor_texture.markrgb = [.36,.41,.43]
    material = next((m for m in spec.materials if m.name=='groundplane'), None)
    if material is not None:
        material.texrepeat = [1,1]
        material.reflectance = 0
    floor = next((g for g in spec.geoms if g.name=='floor' and g.type==mujoco.mjtGeom.mjGEOM_PLANE), None)
    if floor is not None:
        floor.size[:2] = 0
    if not any(t.type==mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        spec.add_texture(name='tracking_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=128,height=768,
                         rgb1=[.48,.59,.66],rgb2=[.83,.86,.85])
    spec.visual.rgba.haze = [.65,.73,.76,1]
    model = spec.compile()
    if (model.nq!=31 or model.njnt!=25 or model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE
            or tuple(model.joint(i).name for i in range(1,25))!=JOINT_NAMES
            or not np.array_equal(model.jnt_qposadr[1:],np.arange(7,31))):
        raise ValueError('Visual model must use the same free root and 24 named hinge joints')
    return model


def load_font(size, weight='regular'):
    names = {'regular':'DejaVuSans.ttf','bold':'DejaVuSans-Bold.ttf','mono':'DejaVuSansMono.ttf'}
    return ImageFont.truetype(names[weight],size)


def draw_path(draw, points, color, width):
    if len(points)>1:
        draw.line([tuple(p) for p in points],fill=color,width=width,joint='curve')


class FrameLayout:
    def __init__(self, trace, title, policy, credit):
        self.trace = trace
        self.base = Image.new('RGB',SIZE,DARK)
        canvas = ImageDraw.Draw(self.base)
        self.medium = load_font(24,'mono')
        self.small = load_font(15)
        self.small_mono = load_font(15,'mono')
        self.label = load_font(14,'bold')
        canvas.text((22,12),'SUPERMIMIC  /  PM01    •    SIMULATION',font=self.label,fill=ACTUAL)
        canvas.text((20,32),title,font=load_font(38,'bold'),fill=TEXT)
        canvas.text((1257,13),'SIMULATION TIME',font=self.label,fill=MUTED,anchor='ra')
        canvas.line((0,86,1280,86),fill=GRID,width=2)
        for left,color,label in ((18,ACTUAL,policy+'  /  Closed-loop rollout'),
                                 (646,REFERENCE,'Motion reference  /  Kinematic replay')):
            canvas.ellipse((left+3,99,left+12,108),fill=color)
            canvas.text((left+21,91),label,font=load_font(18,'bold'),fill=TEXT)
        self.gap = np.linalg.norm(trace.qpos[:,:2]-trace.ref_qpos[:,:2],axis=1)
        mae = np.abs(np.rad2deg(trace.qpos[1:,7:30]-trace.ref_qpos[1:,7:30])).mean()
        for x,label,value in ((21,'ROOT XY GAP',None),(216,'23-JOINT MAE · CLIP',f'{mae:.2f}°'),
                              (468,'CONTROL / PHYSICS',f'{trace.control_hz} / {trace.physics_hz} Hz')):
            canvas.text((x,590),label,font=self.label,fill=MUTED)
            if value is not None:
                canvas.text((x,611),value,font=self.medium,fill=TEXT)
        canvas.text((758,588),'GROUND PATH',font=self.label,fill=MUTED)
        canvas.text((758,615),'XY · metres',font=self.small,fill=TEXT)
        canvas.line((0,666,1280,666),fill=GRID)
        canvas.text((21,683),credit,font=self.small,fill=MUTED)
        canvas.text((1259,683),'ONNX FP32   /   CPU MuJoCo   /   1× playback',
                    font=self.small,fill=MUTED,anchor='ra')
        self.planned_s = trace.planned_steps/trace.control_hz
        left,top,right,bottom = 888,579,1256,655
        paths = np.concatenate((trace.qpos[:,:2],trace.ref_qpos[:,:2]))
        low, high = paths.min(axis=0),paths.max(axis=0)
        region = (np.array([left+7,bottom-7]),np.array([right-7,top+7]))
        width, height = (right-left-14,bottom-top-14)
        # Equal X/Y scales retain angles, distances and drift in world space.
        self.map_scale = min(width/max(.5,high[0]-low[0]),height/max(.35,high[1]-low[1]))
        origin = (region[0]+region[1])/2
        centre = (low+high)/2
        def pixels(xy):
            return np.rint((xy-centre)*[self.map_scale,-self.map_scale]+origin).astype(int)
        self.actual_path,self.ref_path = pixels(trace.qpos[:,:2]),pixels(trace.ref_qpos[:,:2])
        canvas.rectangle((left,top,right,bottom),outline=GRID)
        draw_path(canvas,self.ref_path,'#52606a',1)
        scale_m = max(u for u in (.001,.005,.01,.025,.05,.1,.25,.5,1.,2.,5.,10.) if u*self.map_scale <= 85.)
        x1,y = left+7,bottom-7
        x2 = x1+round(scale_m*self.map_scale)
        canvas.line((x1,y,x2,y),fill=MUTED,width=2)
        canvas.text((x2+4,y-6),f'{scale_m:g} m',font=load_font(11),fill=MUTED,anchor='lm')

    def compose(self, index, actual, reference):
        frame = self.base.copy()
        frame.paste(Image.fromarray(actual),(18,PANEL_Y))
        frame.paste(Image.fromarray(reference),(646,PANEL_Y))
        draw = ImageDraw.Draw(frame)
        elapsed = index/self.trace.control_hz
        draw.text((1257,40),f'{elapsed:05.2f} / {self.planned_s:05.2f} s',font=self.medium,fill=TEXT,anchor='ra')
        draw.line((0,86,round(1279*index/self.trace.planned_steps),86),fill=ACTUAL,width=2)
        draw.text((21,611),f'{self.gap[index]:.2f} m',font=self.medium, fill=TEXT)
        draw_path(draw,self.ref_path[:index+1],REFERENCE,2)
        draw_path(draw,self.actual_path[:index+1],ACTUAL,2)
        for pt,c in ((self.ref_path[index],REFERENCE),(self.actual_path[index],ACTUAL)):
            x,y=pt
            draw.ellipse((x-3,y-3,x+3,y+3),fill=c,outline=DARK)
        if index==len(self.trace.qpos)-1:
            message = 'Reference end' if self.trace.completed else 'Stopped · '+self.trace.stop_reason
            draw.text((1257,70),message,font=self.small,fill=TEXT,anchor='ra')
        return frame


class TrackingRenderer:
    def __init__(self, model):
        self.model = model
        self.state = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model,height=PANE[1],width=PANE[0])
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [1,1,1,0,0,0]
        # The SDK includes diagnostic sites in the default group.
        self.options.sitegroup[:] = 0
        from OpenGL import GL
        self.gl_renderer = GL.glGetString(GL.GL_RENDERER).decode('utf-8')

    def camera(self, trace):
        w,x,y,z = trace.ref_qpos[0,3:7]
        yaw = math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = (yaw+145)%360
        camera.elevation = -15
        camera.lookat[2] = .75
        # Fit the entire clip once, including drift toward the camera. A mesh's
        # local AABB encloses every vertex including hands and toes.
        az, el = np.deg2rad([camera.azimuth, camera.elevation])
        forward = np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
        right = np.array([np.sin(az),-np.cos(az),0])
        up = np.cross(right,forward)
        tangent = math.tan(math.radians(self.model.vis.global_.fovy)/2)
        meshes = np.flatnonzero(self.model.geom_type==mujoco.mjtGeom.mjGEOM_MESH)
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        boxes = self.model.geom_aabb[meshes]
        corners = boxes[:,None,:3] + signs[None,:,:]*boxes[:,None,3:]
        distance = 2.42
        for i, poses in enumerate(zip(trace.qpos,trace.ref_qpos)):
            centre = np.r_[.5*(trace.qpos[i,:2]+trace.ref_qpos[i,:2]),camera.lookat[2]]
            for pose in poses:
                self.state.qpos[:] = pose
                mujoco.mj_kinematics(self.model,self.state)
                rotation = self.state.geom_xmat[meshes].reshape(-1,3,3)
                relative = (np.einsum('mij,mvj->mvi',rotation,corners) +
                            self.state.geom_xpos[meshes,None,:] - centre).reshape(-1,3)
                x_width = np.abs(relative@right)/(PANE[0]/PANE[1])
                y_height = np.abs(relative@up)
                required = np.maximum(x_width,y_height)/(.89*tangent) - relative@forward
                distance = max(distance,float(required.max()))
        camera.distance = distance
        return camera

    def project(self, pose, camera):
        self.state.qpos[:] = pose
        mujoco.mj_forward(self.model,self.state)
        self.renderer.update_scene(self.state,camera=camera,scene_option=self.options)
        return self.renderer.render()

    def frame(self, trace, index, camera, layout):
        # Both panes share the very same camera and the same world-floor origin.
        # The shared midpoint keeps a drifting robot in view without aligning roots.
        camera.lookat[:2] = .5*(trace.qpos[index,:2]+trace.ref_qpos[index,:2])
        actual = self.project(trace.qpos[index],camera)
        reference = self.project(trace.ref_qpos[index],camera)
        return layout.compose(index,actual,reference)

    def close(self):
        self.renderer.close()


def render_video(projector, trace, title, policy, credit, dest, ffmpeg):
    if dest.exists():
        raise FileExistsError(dest)
    layout = FrameLayout(trace,title,policy,credit)
    camera = projector.camera(trace)
    dest.parent.mkdir(parents=True,exist_ok=True)
    fd = tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.mp4',prefix='.render-',delete=False)
    fd.close()
    staging = Path(fd.name)
    command = [ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
               '-video_size',f'{SIZE[0]}x{SIZE[1]}','-framerate',str(trace.control_hz),'-i','pipe:0',
               '-an','-map_metadata','-1','-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt709',
               '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-color_primaries','bt709',
               '-color_trc','bt709','-colorspace','bt709','-threads','2','-movflags','+faststart',str(staging)]
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
            try:
                for i in range(len(trace.qpos)):
                    process.stdin.write(projector.frame(trace,i,camera,layout).tobytes())
            finally:
                process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError('ffmpeg did not complete')
        staging.replace(dest)
    finally:
        staging.unlink(missing_ok=True)
    return dict(samples=len(trace.qpos),fps=trace.control_hz,media_seconds=len(trace.qpos)/trace.control_hz,
                gl_renderer=projector.gl_renderer,azimuth_deg=camera.azimuth,elevation_deg=camera.elevation,
                distance_m=camera.distance,lookat_height_m=camera.lookat[2],
                camera_xy='shared exact midpoint of the policy and reference roots',motion_speed=1,
                view='two synchronized perspective panes',resolution=list(SIZE),encoding='H.264, yuv420p, BT.709, CRF 18')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--visual-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--title',required=True)
    parser.add_argument('--credit',required=True,help='Visible source credit for the recorded motion')
    parser.add_argument('--policy-label',default='65k policy')
    parser.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    parser.add_argument('--require-software-renderer',action='store_true')
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('Provide an ffmpeg executable with --ffmpeg or PATH')
    trace = Trace.load(args.trace)
    projector = TrackingRenderer(load_visual_model(args.visual_model))
    try:
        if args.require_software_renderer and not any(token in projector.gl_renderer.lower()
                                                      for token in ('llvmpipe','softpipe','swrast','software')):
            raise RuntimeError('The active OpenGL renderer is '+projector.gl_renderer)
        info = render_video(projector,trace,args.title,args.policy_label,args.credit,args.output,args.ffmpeg)
        print(json.dumps(info,indent=2))
    finally:
        projector.close()


if __name__=='__main__':
    main()

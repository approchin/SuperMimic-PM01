# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Exact reset-time qpos0 inertia constants for a single moving PM01 torso COM."""
import mujoco
import numpy as np
import torch


class TorsoConstants:
    def __init__(self, model, torso, device):
        self.torso = torso
        self.mass = float(model.body_mass[torso])
        if model.ntendon or model.neq or model.nv != 30:
            raise ValueError("Torso inertia specialization requires the native free-base PM01 model")
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        mat = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mat)
        jac = np.zeros((model.nbody, 2, 3, model.nv))
        for b in range(1, model.nbody):
            mujoco.mj_jacBodyCom(model, data, jac[b, 0], jac[b, 1], b)
        tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), device=device, dtype=torch.float64)
        self.base_mass = tensor(mat)
        self.translation = tensor(jac[torso, 0])
        self.angular = tensor(jac[torso, 1])
        self.rotation = tensor(data.xmat[torso].reshape(3, 3))
        # trace(J inv(M) J^T)/3 for each body and each of translation/rotation.
        self.body_forms = tensor(np.einsum("btij,btik->btjk", jac,jac)/3).reshape(model.nbody*2,-1).T.contiguous()
        self.drive_dofs = torch.as_tensor(model.jnt_dofadr[model.actuator_trnid[:, 0]].copy(), device=device,dtype=torch.long)
        self.drive_gear = tensor(model.actuator_gear[:, 0])

    def compute(self, offsets):
        delta_world = offsets.double() @ self.rotation.T
        jac = self.translation + torch.linalg.cross(self.angular.T[None],delta_world[:,None,:],dim=-1).transpose(1,2)
        delta_mass = self.mass * (jac.transpose(1,2)@jac-self.translation.T@self.translation)
        mass = self.base_mass + delta_mass
        inv = torch.linalg.inv_ex(mass)[0]
        dof = inv.diagonal(dim1=-2,dim2=-1).clone()
        for start in (0,3):
            dof[:,start:start+3] = dof[:,start:start+3].mean(-1,keepdim=True)
        body = (inv.flatten(1)@self.body_forms).reshape(len(inv),-1,2)
        body[:,self.torso,0] = ((jac@inv)*jac).sum((1,2))/3
        acc = (inv[:,:,self.drive_dofs]*self.drive_gear).norm(dim=1)
        mean_inertia = mass.diagonal(dim1=-2,dim2=-1).mean(-1)
        return {k:v.float() for k,v in (("body_invweight0",body),("dof_invweight0",dof),("actuator_acc0",acc),("meaninertia",mean_inertia))}

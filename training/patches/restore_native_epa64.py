# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Restore the verified 004/010 EPA capacity in this run's private 3.16 wheel."""
import argparse
import hashlib
from importlib import metadata
import os
from pathlib import Path
import sys

STOCK = "655f6a0ee10794e09205289ebb682c0a281a93c08d33348530905da5f7d31962"
PINS = {
    "types.py": "78752384be7078eb873a3d55105db4abd0e8f715c898e32c47eb35fff272710d",
    "collision_gjk.py": "c288f8f37e4b7f8f9e0e48a92a46c3eee0d0efc6dd7fac65bba2522e11f39e48",
    "collision_convex.py": "3fd288f0cab3f5f444924d724362fd2cd2eab1281ab2dcb6bc3b48b9b3f30b92",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(root, check):
    private = (root/".venv").resolve()
    if Path(sys.prefix).resolve() != private or metadata.version("mujoco-warp") != "3.16.0":
        raise ValueError("Use the named run's private 3.16.0 Python environment")
    src = Path(metadata.distribution("mujoco-warp").locate_file("mujoco_warp/_src")).resolve()
    if not src.is_relative_to(private):
        raise ValueError("The selected package is outside the run's private environment")
    for name in ("collision_gjk.py","collision_convex.py"):
        if digest((src/name).read_bytes()) != PINS[name]:
            raise ValueError("Unexpected collision implementation: "+name)
    path=src/"types.py"; original=path.read_bytes()
    if digest(original) == PINS["types.py"]:
        print("Verified native EPA64 closure: all three pinned files already match.",flush=True)
        return
    before,after=b"MJ_MAX_EPAHORIZON = 24",b"MJ_MAX_EPAHORIZON = 64"
    fixed=original.replace(before,after)
    if check or original.count(before)!=1 or digest(original)!=STOCK or digest(fixed)!=PINS["types.py"]:
        raise ValueError("Native EPA64 source contract mismatch")
    tmp=src/("types.py.epa64."+str(os.getpid())+".tmp")
    try:
        with tmp.open("xb") as f: f.write(fixed)
        tmp.chmod(path.stat().st_mode)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)
    for f in (src/"__pycache__").glob("types.cpython-*.pyc"):
        f.unlink()
    print("Restored native EPA64 in this private environment.",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--check",action="store_true")
    args=p.parse_args();main(args.root,args.check)

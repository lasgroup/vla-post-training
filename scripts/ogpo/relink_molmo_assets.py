"""Rewrite absolute symlinks under <mlspaces>/assets that point into <mlspaces> as relative ones, so the tree can be
tarred and unpacked anywhere (MolmoSpaces installs assets/ as symlinks into cache/ with the absolute paths of the host).

    python3 relink.py /data/user_data/sarveshp/molmo_stage/mlspaces
"""
import os
import sys

root = os.path.realpath(sys.argv[1])
n_rel = n_other = 0
for base, dirs, files in os.walk(os.path.join(root, "assets")):
    for name in dirs + files:
        path = os.path.join(base, name)
        if not os.path.islink(path):
            continue
        target = os.readlink(path)
        if not os.path.isabs(target):
            continue
        real = os.path.realpath(target) if os.path.lexists(target) else target
        inside = real.startswith(root + os.sep)
        if not inside and "/molmo_stage/mlspaces/" in target:  # e.g. /usr0/data/... spelling of the same dir
            real = root + target.split("/molmo_stage/mlspaces", 1)[1]
            inside = True
        if not inside:
            n_other += 1
            print("outside link kept:", path, "->", target)
            continue
        rel = os.path.relpath(real, os.path.dirname(path))
        os.remove(path)
        os.symlink(rel, path)
        n_rel += 1
print(f"relinked {n_rel} symlinks, {n_other} left absolute")
bad = [os.path.join(b, n) for b, ds, fs in os.walk(os.path.join(root, "assets")) for n in ds + fs
       if os.path.islink(os.path.join(b, n)) and not os.path.exists(os.path.join(b, n))]
print(f"dangling links after relink: {len(bad)}", bad[:3])

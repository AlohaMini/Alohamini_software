"""用 Git 原生增量包同步已提交源码，不触碰机器人工作空间。"""

import argparse
import base64
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def clean_repository(root):
    root = Path(git(root, "rev-parse", "--show-toplevel"))
    if git(root, "status", "--porcelain"):
        raise ValueError("源码有未提交修改，请先保存修改后再同步")
    return root


def prepare_bundle(root, base, destination):
    from alohamini.software_version import _source_sha256

    root = clean_repository(root)
    commit = git(root, "rev-parse", "HEAD")
    branch = git(root, "branch", "--show-current")
    if not branch:
        raise ValueError("电脑源码未处于分支，不能同步到机器人")
    if branch == "main":
        raise ValueError("当前同步流程仅用于开发分支，main 保持不动")
    git(root, "merge-base", "--is-ancestor", base, commit)
    if commit != base:
        git(root, "bundle", "create", str(destination), "HEAD", "^" + base)
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("准备期间电脑代码已变化，请重新同步")
    clean_repository(root)
    digest = _source_sha256(root / "src/alohamini")
    if digest is None:
        raise ValueError("无法读取电脑源码指纹")
    if commit != base and Path(destination).stat().st_size > 5 * 1024 * 1024:
        raise ValueError("增量包超过 5 MiB，请先同步基础版本后再使用一键同步")
    return dict(commit=commit, branch=branch, base=base, source_sha256=digest,
                bundle=base64.b64encode(Path(destination).read_bytes()).decode("ascii") if commit != base else "")


def apply_bundle(root, package, directory):
    root = clean_repository(root)
    previous = dict(commit=git(root, "rev-parse", "HEAD"), branch=git(root, "branch", "--show-current"))
    if previous["commit"] != package["base"]:
        raise ValueError("机器人代码已变化，请重新检查版本后同步")
    git(root, "check-ref-format", "--branch", package["branch"])
    if package["bundle"]:
        bundle = Path(directory) / "software.bundle"
        bundle.write_bytes(base64.b64decode(package["bundle"], validate=True))
        git(root, "bundle", "verify", str(bundle))
        if git(root, "bundle", "list-heads", str(bundle)) != package["commit"] + " HEAD":
            raise ValueError("同步包的提交与电脑版本不一致")
        git(root, "fetch", str(bundle), "HEAD")
    git(root, "merge-base", "--is-ancestor", previous["commit"], package["commit"])
    existing = subprocess.run(["git", "-C", str(root), "show-ref", "--verify", "--quiet",
                               "refs/heads/" + package["branch"]], timeout=5)
    if existing.returncode == 0:
        if package["branch"] != previous["branch"] and git(root, "rev-parse", package["branch"]) != previous["commit"]:
            raise ValueError("机器人已有同名开发分支且提交不同，请先保存该分支")
        git(root, "merge-base", "--is-ancestor", package["branch"], package["commit"])
        git(root, "switch", package["branch"])
        git(root, "merge", "--ff-only", package["commit"])
    if existing.returncode != 0:
        git(root, "switch", "-c", package["branch"], package["commit"])
    return previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.root, args.base, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()

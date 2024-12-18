from loguru import logger
import subprocess


def clone_repo(repo: str, local_dir: str) -> None:
    logger.debug(f"cloning {repo}")
    repo_url = f"https://github.com/{repo}"
    subprocess.run(["git", "clone", repo_url, local_dir], check=True)

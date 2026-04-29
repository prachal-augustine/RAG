import subprocess

packages = [p.split('==')[0] for p in subprocess.getoutput('pip freeze').splitlines() if p.startswith('langgraph')]
for pkg in packages:
    subprocess.run(['pip', 'uninstall', '-y', pkg])
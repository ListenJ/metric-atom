"""Upload MetricAtom project to remote server."""
import paramiko, os, sys

SSH_HOST = "region-9.autodl.pro"
SSH_PORT = 20649
SSH_USER = "root"
SSH_PASS = "Sa+b7b3GsqG6"

LOCAL = r"D:\MetricAtom"
REMOTE = "/root/MetricAtom"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
sftp = ssh.open_sftp()

# Collect all files to upload
files_to_upload = []

def add_files(local_dir, remote_dir, exts={".py", ".txt", ".sh", ".md", ".gitignore"}):
    for root, dirs, files in os.walk(local_dir):
        basename = os.path.basename(root)
        if basename.startswith(".") or basename in ("__pycache__", "node_modules", "outputs", "outputs_remote", ".venv", ".git", ".idea", ".claude", ".omc", ".opencode", ".mypy_cache", ".pytest_cache"):
            continue
        rel = os.path.relpath(root, local_dir)
        if rel == ".":
            remote_root = remote_dir
        else:
            remote_root = remote_dir + "/" + rel.replace("\\", "/")
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext in exts or fname in (".gitignore", "requirements.txt", "project_plan.md"):
                local_path = os.path.join(root, fname)
                remote_path = remote_root + "/" + fname
                files_to_upload.append((local_path, remote_path))

# Source code and key files
add_files(os.path.join(LOCAL, "src"), REMOTE + "/src")
add_files(os.path.join(LOCAL, "tests"), REMOTE + "/tests")
add_files(os.path.join(LOCAL, "scripts"), REMOTE + "/scripts")
add_files(LOCAL, REMOTE, exts={".py", ".txt", ".sh", ".md", ".gitignore", ".ini"})

# Create remotedirs first
remote_dirs = set()
for _, rp in files_to_upload:
    rd = rp.rsplit("/", 1)[0]
    remote_dirs.add(rd)

for rd in sorted(remote_dirs):
    try:
        ssh.exec_command("mkdir -p " + rd)
    except:
        pass

print("Created remote directories. Starting upload...")

success = 0
for local, remote in files_to_upload:
    try:
        sftp.put(local, remote)
        success += 1
    except Exception as e:
        print(f"  FAIL: {local} -> {remote}: {e}")

print(f"\nUploaded {success} files")

# Verify
stdin, stdout, _ = ssh.exec_command("find " + REMOTE + " -type f | wc -l")
print(f"Remote file count: {stdout.read().decode().strip()}")

sftp.close()
ssh.close()

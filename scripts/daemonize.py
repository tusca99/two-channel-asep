#!/usr/bin/env python3
"""Double-fork daemonizer: launch a shell command line fully detached."""
import os, subprocess, sys

shell_cmd = " ".join(sys.argv[1:])
pid1 = os.fork()
if pid1 == 0:
    os.setsid()
    pid2 = os.fork()
    if pid2 == 0:  # grandchild: reparented to init, survives parent exit
        dn = open(os.devnull, 'r+')
        subprocess.Popen(["/bin/bash", "-c", shell_cmd],
                         cwd="/home/user_vr/Documents/two-channel-asep",
                         stdin=dn, stdout=dn, stderr=dn, close_fds=True)
        os._exit(0)
    os._exit(0)
os.waitpid(pid1, 0)
print("daemonized")

#!/bin/bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "started container"
/usr/sbin/sshd -D

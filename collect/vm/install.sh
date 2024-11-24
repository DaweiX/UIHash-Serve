#!/bin/sh

set -xe

qemu-img create -f qcow2 "$2" 8G

qemu-system-x86_64 -vga std \
                   -m 4096 -smp 2 \
                   -net nic,model=e1000 -net user \
                   -cdrom "$1" \
                   -hda "$2" \
                   -boot d \
                   -nographic

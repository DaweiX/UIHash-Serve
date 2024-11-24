#!/bin/sh

set -xe

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

qemu-system-x86_64 -enable-kvm -vga std \
                   -m 2048 -smp 2 -cpu host \
                   -device e1000,netdev=net0 \
                   -netdev user,id=net0,hostfwd=tcp::5555-:5555 \
                   -hda "$SCRIPT_DIR/vm/android.img" \
                   -nographic \
                   "$@"


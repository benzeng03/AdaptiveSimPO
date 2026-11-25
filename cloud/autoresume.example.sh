#!/bin/bash
# Copy to ~/autoresume.sh on the VM and set RUN; cloud/startup-script.sh runs it at every boot,
# so a preempted or rebuilt VM resumes the run in tmux session "simpo" by itself.
# ~/start_full.sh is the launcher: it runs cloud/run.sh with the experiment's options and
# passes its first argument as --run-dir (see cloud/GUIDE.zh.md, step 7).
RUN=outputs/run-XXXX
cd ~/AdaptiveSimPO || exit 1
if [ -f "$RUN.tar.gz" ]; then echo "[autoresume] $RUN finished, nothing to do"; exit 0; fi
for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done   # wait for the GPU driver
if tmux has-session -t simpo 2>/dev/null; then echo "[autoresume] tmux session exists"; exit 0; fi
tmux new-session -d -s simpo bash ~/start_full.sh "$RUN"
echo "[autoresume] $(date '+%F %T') resumed $RUN in tmux session simpo" >> ~/autoresume.log

#!/bin/bash
# GCE startup script (runs as root on every boot). For each user whose home has an
# executable ~/autoresume.sh, run it as that user, so a preempted or rebuilt VM
# picks its experiment up again without anyone logging in. Install with
#   gcloud compute instances add-metadata simpo-l4 --zone ZONE --metadata-from-file startup-script=cloud/startup-script.sh
# and see cloud/GUIDE.zh.md (step 7) for ~/autoresume.sh.
for f in /home/*/autoresume.sh; do
  [ -f "$f" ] || continue
  user=$(stat -c %U "$f")
  echo "[startup] running $f as $user"
  sudo -u "$user" -i bash "$f" &
done
wait

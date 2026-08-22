#!/bin/bash
# Show who is using a node, how many GPUs each of them holds, and on which partitions.
# Usage: ./node_usage.sh [node]   (default: babel-m9-16)

NODE=${1:-babel-m9-16}

INFO=$(scontrol show node "$NODE" 2>/dev/null) || { echo "unknown node: $NODE" >&2; exit 1; }
[ -z "$INFO" ] && { echo "unknown node: $NODE" >&2; exit 1; }

echo "=== $NODE ==="
echo "  State      = $(grep -oP 'State=\K\S+' <<<"$INFO")"
echo "  Partitions = $(grep -oP 'Partitions=\K\S+' <<<"$INFO")"
echo "  GPUs       = $(grep -oP 'CfgTRES=.*gres/gpu=\K[0-9]+' <<<"$INFO") total, $(grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' <<<"$INFO" || echo 0) allocated"
echo "  Type       = $(grep -oP 'Gres=gpu:\K[^:]+' <<<"$INFO")"
echo

TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

# squeue's tres-per-node is job-wide and gets truncated, so per-node GPU counts
# come from the detailed "Nodes=<node> ... GRES=gpu:TYPE:N" block of each job.
DETAIL=$(scontrol -d show job 2>/dev/null)

printf "%-11s %-11s %-11s %4s  %-11s %s\n" JOBID USER PARTITION GPUS TIME NAME
while IFS='|' read -r jobid user part state time name; do
    [ "$state" = "RUNNING" ] || continue
    gpus=$(awk -v id="$jobid" -v node="$NODE" '
        /^JobId=/          { cur = ($1 == "JobId=" id) }
        cur && /Nodes=/ && $0 ~ ("Nodes=" node "([ ,]|$)") {
            if (match($0, /GRES=gpu:[A-Za-z_0-9]+:[0-9]+/)) {
                s = substr($0, RSTART, RLENGTH); sub(/.*:/, "", s); print s; exit
            }
        }' <<<"$DETAIL")
    gpus=${gpus:-0}
    printf "%-11s %-11s %-11s %4s  %-11s %s\n" "$jobid" "$user" "$part" "$gpus" "$time" "$name"
    printf "%s\t%s\t%s\n" "$user" "$gpus" "$part" >> "$TMP"
done < <(squeue -h -w "$NODE" -O "JobID:|,UserName:|,Partition:|,State:|,TimeUsed:|,Name:|" | sed 's/ *|/|/g')

echo
printf "%-11s %4s %5s  %s\n" USER GPUS JOBS PARTITIONS
sort "$TMP" | awk -F'\t' '
    { g[$1] += $2; n[$1]++; if (index(p[$1], $3) == 0) p[$1] = p[$1] (p[$1] ? "," : "") $3 }
    END { for (u in g) printf "%-11s %4d %5d  %s\n", u, g[u], n[u], p[u] }' | sort -k2 -rn

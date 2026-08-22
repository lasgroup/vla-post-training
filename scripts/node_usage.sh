#!/bin/bash
# Show who is using the maxlab nodes, how many GPUs each of them holds, and on
# which partitions.
# Usage: ./node_usage.sh [node ...]   (default: every node in the maxlab partition)

PARTITION=${PARTITION:-maxlab}

if [ $# -gt 0 ]; then
    NODES=("$@")
else
    mapfile -t NODES < <(sinfo -h -p "$PARTITION" -N -o "%N" | sort -u)
fi
[ ${#NODES[@]} -eq 0 ] && { echo "no nodes in partition $PARTITION" >&2; exit 1; }

# squeue's tres-per-node is job-wide and gets truncated, so per-node GPU counts
# come from the detailed "Nodes=<node> ... GRES=gpu:TYPE:N" block of each job.
# One scontrol call covers every node below.
DETAIL=$(scontrol -d show job 2>/dev/null)

ALL=$(mktemp); trap 'rm -f "$ALL"' EXIT

for NODE in "${NODES[@]}"; do
    INFO=$(scontrol show node "$NODE" 2>/dev/null)
    if [ -z "$INFO" ]; then
        echo "unknown node: $NODE" >&2
        continue
    fi

    gpu_tot=$(grep -oP 'CfgTRES=.*?gres/gpu=\K[0-9]+' <<<"$INFO")
    gpu_use=$(grep -oP 'AllocTRES=.*?gres/gpu=\K[0-9]+' <<<"$INFO")

    echo "=== $NODE ==="
    echo "  State      = $(grep -oP 'State=\K\S+' <<<"$INFO")"
    echo "  Partitions = $(grep -oP 'Partitions=\K\S+' <<<"$INFO")"
    echo "  GPUs       = ${gpu_use:-0}/${gpu_tot:-0} allocated ($(grep -oP 'Gres=gpu:\K[^:]+' <<<"$INFO"))"
    echo

    NODE_JOBS=$(mktemp)
    printf "  %-11s %-11s %-11s %4s  %-11s %s\n" JOBID USER PARTITION GPUS TIME NAME
    while IFS='|' read -r jobid user part state time name; do
        [ "$state" = "RUNNING" ] || continue
        gpus=$(awk -v id="$jobid" -v node="$NODE" '
            /^JobId=/ { cur = ($1 == "JobId=" id) }
            cur && $0 ~ ("Nodes=" node "([ ,]|$)") {
                if (match($0, /GRES=gpu:[A-Za-z_0-9]+:[0-9]+/)) {
                    s = substr($0, RSTART, RLENGTH); sub(/.*:/, "", s); print s; exit
                }
            }' <<<"$DETAIL")
        gpus=${gpus:-0}
        printf "  %-11s %-11s %-11s %4s  %-11s %s\n" "$jobid" "$user" "$part" "$gpus" "$time" "$name"
        printf "%s\t%s\t%s\n" "$user" "$gpus" "$part" | tee -a "$ALL" >> "$NODE_JOBS"
    done < <(squeue -h -w "$NODE" -O "JobID:|,UserName:|,Partition:|,State:|,TimeUsed:|,Name:|" | sed 's/ *|/|/g')

    echo
    printf "  %-11s %4s %5s  %s\n" USER GPUS JOBS PARTITIONS
    sort "$NODE_JOBS" | awk -F'\t' '
        { g[$1] += $2; n[$1]++; if (index(p[$1], $3) == 0) p[$1] = p[$1] (p[$1] ? "," : "") $3 }
        END { for (u in g) printf "  %-11s %4d %5d  %s\n", u, g[u], n[u], p[u] }' | sort -k2 -rn
    rm -f "$NODE_JOBS"
    echo
done

if [ ${#NODES[@]} -gt 1 ]; then
    echo "=== all ${#NODES[@]} nodes ==="
    printf "  %-11s %4s %5s  %s\n" USER GPUS JOBS PARTITIONS
    sort "$ALL" | awk -F'\t' '
        { g[$1] += $2; n[$1]++; if (index(p[$1], $3) == 0) p[$1] = p[$1] (p[$1] ? "," : "") $3 }
        END { for (u in g) printf "  %-11s %4d %5d  %s\n", u, g[u], n[u], p[u] }' | sort -k2 -rn
fi

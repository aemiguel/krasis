#!/usr/bin/env bash
set -euo pipefail

cd /home/main/Documents/Claude/krasis

log_path="logs/manual/phase2br_qcn_hqq_speed_matrix_20260427.log"
{
  printf "PHASE2BR QCN HQQ speed matrix start %s\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
  ./dev kill || true

  printf "\n[1/2] HQQ4SC speed benchmark\n"
  ./dev benchmark tests/qcn-polar4-hqq4sc.conf
  ./dev kill || true

  printf "\n[2/2] HQQ8 speed benchmark\n"
  ./dev benchmark tests/qcn-polar4-hqq8.conf
  ./dev kill || true

  printf "PHASE2BR QCN HQQ speed matrix end %s\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
} > "${log_path}" 2>&1

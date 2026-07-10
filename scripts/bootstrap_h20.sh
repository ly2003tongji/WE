#!/usr/bin/env bash

set -uo pipefail

RESEARCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${RESEARCH_ROOT}/upstream/WorldEngine"
REPORT_DIR="${RESEARCH_ROOT}/reports"
REPORT_PATH="${REPORT_DIR}/bootstrap_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "${REPORT_DIR}" "${RESEARCH_ROOT}/upstream"

run_optional() {
  local label="$1"
  shift
  echo
  echo "## ${label}"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || true
  else
    echo "未找到命令: $1"
  fi
}

{
  echo "# H20 Bootstrap Audit"
  echo "time=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "research_root=${RESEARCH_ROOT}"

  run_optional "系统" uname -a
  run_optional "操作系统" bash -c 'test -f /etc/os-release && source /etc/os-release && echo "${PRETTY_NAME}"'
  run_optional "CPU" lscpu
  run_optional "内存" free -h
  run_optional "GPU" nvidia-smi
  run_optional "Python" python3 --version
  run_optional "Conda" conda --version
  run_optional "Git" git --version

  echo
  echo "## 文件系统"
  df -h "${RESEARCH_ROOT}" 2>&1 || true
  if [[ -d /mnt/cpfs ]]; then
    df -h /mnt/cpfs 2>&1 || true
  else
    echo "/mnt/cpfs 不存在"
  fi

  echo
  echo "## 官方仓库"
  if [[ -d "${UPSTREAM_DIR}/.git" ]]; then
    echo "官方仓库已存在，执行 fast-forward only 更新"
    git -C "${UPSTREAM_DIR}" pull --ff-only 2>&1 || true
  else
    git clone https://github.com/OpenDriveLab/WorldEngine.git "${UPSTREAM_DIR}" 2>&1 || true
  fi

  if [[ -d "${UPSTREAM_DIR}/.git" ]]; then
    git -C "${UPSTREAM_DIR}" status --short --branch 2>&1 || true
    git -C "${UPSTREAM_DIR}" log -1 --format='commit=%H%ncommit_date=%cI%nsubject=%s' 2>&1 || true
  fi

  echo
  echo "## 安全停止点"
  echo "尚未安装环境、编译 CUDA 扩展或下载数据。"
  echo "下一步必须先审计官方数据 manifest 和体积。"
} | tee "${REPORT_PATH}"

echo
echo "审计报告已保存：${REPORT_PATH}"


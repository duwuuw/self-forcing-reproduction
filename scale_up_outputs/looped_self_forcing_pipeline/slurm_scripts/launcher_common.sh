#!/usr/bin/env bash

_is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

_is_nonnegative_integer() {
  [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]
}

_is_bundle_relative_path() {
  local value=$1
  [[ -n "$value" && "$value" != /* && "$value" != '~'* && "$value" != *\\* ]] || return 1
  [[ "$value" =~ ^[A-Za-z0-9._/-]+$ ]] || return 1
  case "/$value/" in
    *"/../"*|*"/./"*) return 1 ;;
  esac
  return 0
}

resolve_logs_root() {
  local scripts_root=$1
  local exp_dir=$2
  local logs_root

  if ! logs_root="$("$SUE_PYTHON" - "$scripts_root" "$exp_dir" 2>/dev/null <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from looped_self_forcing_pipeline.config import resolve_runtime_paths

print(resolve_runtime_paths(sys.argv[2])["logs_root"])
PY
)"; then
    printf 'unable to resolve logs_root from the selected runtime config\n' >&2
    return 2
  fi
  if [[ -z "$logs_root" || "$logs_root" != /* ]]; then
    printf 'runtime config did not resolve logs_root to an absolute path\n' >&2
    return 2
  fi
  printf '%s\n' "$logs_root"
}

runtime_environment_value() {
  local runtime_file=$1
  local key=$2
  [[ -r "$runtime_file" && "$key" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] || {
    printf 'runtime environment key is unavailable\n' >&2
    return 2
  }
  awk -v key="$key" '
    /^environment:[[:space:]]*$/ { in_environment = 1; next }
    in_environment && /^[^[:space:]]/ { exit }
    in_environment && $1 == key ":" {
      sub(/^[[:space:]]*[^:]+:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      gsub(/^\047|\047$/, "")
      print
      found = 1
      exit
    }
    END { if (!found) exit 1 }
  ' "$runtime_file"
}

configure_nm5_runtime_environment() {
  local workspace_root=$1
  local exp_dir=$2
  local runtime_file="$exp_dir/config/runtime.yaml"
  local python_binary_relative asset_root_relative python_overlay_relative python_overlay relative_path
  [[ -r "$runtime_file" ]] || {
    printf 'selected NM5 runtime config is unavailable\n' >&2
    return 2
  }
  python_binary_relative="$(runtime_environment_value "$runtime_file" python_binary)" || {
    printf 'runtime environment.python_binary is required for NM5\n' >&2
    return 2
  }
  asset_root_relative="$(runtime_environment_value "$runtime_file" asset_root)" || {
    printf 'runtime environment.asset_root is required for NM5\n' >&2
    return 2
  }
  for relative_path in "$python_binary_relative" "$asset_root_relative"; do
    _is_bundle_relative_path "$relative_path" || {
      printf 'runtime NM5 environment paths must be safe relative paths\n' >&2
      return 2
    }
  done

  SUE_PYTHON="${SUE_PYTHON:-$workspace_root/$python_binary_relative}"
  [[ -x "$SUE_PYTHON" ]] || {
    printf 'configured NM5 Python interpreter is unavailable\n' >&2
    return 2
  }
  SUE_PYTHON="$(realpath -e -- "$SUE_PYTHON")"
  if [[ -z "${SUE_ASSET_ROOT:-}" ]]; then
    SUE_ASSET_ROOT="$workspace_root/$asset_root_relative"
  fi
  [[ "$SUE_ASSET_ROOT" == /* && -d "$SUE_ASSET_ROOT" ]] || {
    printf 'configured NM5 asset root is unavailable\n' >&2
    return 2
  }
  SUE_ASSET_ROOT="$(realpath -e -- "$SUE_ASSET_ROOT")"
  [[ -d "$SUE_ASSET_ROOT/checkpoints" && -d "$SUE_ASSET_ROOT/wan_models" ]] || {
    printf 'configured NM5 asset root lacks expected checkpoint and model directories\n' >&2
    return 2
  }

  SUE_PYTHONPATH=""
  python_overlay_relative="$(runtime_environment_value "$runtime_file" python_overlay 2>/dev/null || true)"
  if [[ -n "$python_overlay_relative" ]]; then
    _is_bundle_relative_path "$python_overlay_relative" || {
      printf 'runtime environment.python_overlay must be bundle-relative\n' >&2
      return 2
    }
    python_overlay="$(realpath -e -- "$exp_dir/$python_overlay_relative" 2>/dev/null)" || {
      printf 'configured NM5 Python overlay is unavailable\n' >&2
      return 2
    }
    [[ -d "$python_overlay" && "$python_overlay" == "$exp_dir"/* ]] || {
      printf 'configured NM5 Python overlay must stay inside SUE_EXP_DIR\n' >&2
      return 2
    }
    SUE_PYTHONPATH="$python_overlay"
  fi
  PYTHONPATH="$SUE_PYTHONPATH"
  export SUE_PYTHON SUE_ASSET_ROOT SUE_PYTHONPATH PYTHONPATH
}

load_nm5_cache_environment() {
  local exp_dir=$1
  local workspace_root=$2
  local values line target source_name cache_path resolved home_resolved
  local count=0
  local -a export_names=()
  values="$("$SUE_PYTHON" - "$exp_dir" "$workspace_root" 2>/dev/null <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
from looped_self_forcing_pipeline.config import load_runtime_config

runtime = load_runtime_config(sys.argv[1])
mapping = runtime.get("environment", {}).get("cache_env_sources", {})
for target, source_name in mapping.items():
    value = os.environ.get(source_name, "").strip()
    if not value:
        raise SystemExit(2)
    print(f"{target}\t{source_name}\t{value}")
PY
)" || {
    printf 'required NM5 cache roots are missing or runtime cache mapping is invalid\n' >&2
    return 2
  }
  home_resolved=""
  if [[ -n "${HOME:-}" ]]; then
    home_resolved="$(realpath -e -- "$HOME" 2>/dev/null || true)"
  fi
  while IFS=$'\t' read -r target source_name cache_path; do
    [[ -n "$target" && -n "$source_name" && -n "$cache_path" ]] || {
      printf 'NM5 runtime cache mapping is incomplete\n' >&2
      return 2
    }
    [[ "$target" =~ ^(HF_HOME|HF_HUB_CACHE|MODELSCOPE_CACHE|TORCH_HOME|MPLCONFIGDIR|WANDB_CACHE_DIR)$ &&
       "$source_name" =~ ^NM5_[A-Z0-9_]+$ && "$cache_path" == /* &&
       "$cache_path" != *$'\n'* && "$cache_path" != *$'\t'* ]] || {
      printf 'NM5 cache roots must be absolute private backend paths\n' >&2
      return 2
    }
    [[ -d "$cache_path" && -w "$cache_path" ]] || {
      printf 'a required NM5 cache root is unavailable or not writable\n' >&2
      return 2
    }
    resolved="$(realpath -e -- "$cache_path")"
    if [[ -n "$home_resolved" && ( "$resolved" == "$home_resolved" || "$resolved" == "$home_resolved"/* ) ]]; then
      printf 'NM5 cache roots must stay outside the home directory\n' >&2
      return 2
    fi
    printf -v "$target" '%s' "$resolved"
    export "$target"
    export_names+=("$target")
    ((count += 1))
  done <<<"$values"
  [[ "$count" == 6 ]] || {
    printf 'runtime must provide all six NM5 cache roots\n' >&2
    return 2
  }
  printf -v SUE_NM5_CACHE_EXPORTS '%s,' "${export_names[@]}"
  SUE_NM5_CACHE_EXPORTS="${SUE_NM5_CACHE_EXPORTS%,}"
}

slurm_duration_seconds() {
  local value=$1 days=0 hours minutes seconds
  if [[ "$value" == *-* ]]; then
    days=${value%%-*}
    value=${value#*-}
    _is_nonnegative_integer "$days" || return 2
  fi
  IFS=: read -r hours minutes seconds <<<"$value"
  [[ "$hours" =~ ^[0-9]{1,2}$ && "$minutes" =~ ^[0-9]{1,2}$ && "$seconds" =~ ^[0-9]{1,2}$ ]] || return 2
  (( 10#$minutes < 60 && 10#$seconds < 60 && 10#$hours < 24 )) || return 2
  printf '%s\n' "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

require_nm5_partition_time() {
  local partition=$1 requested_seconds=$2 info max_time max_seconds days remaining hours minutes seconds
  [[ "$requested_seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Hydra timeout_seconds must be a positive integer\n' >&2
    return 2
  }
  command -v scontrol >/dev/null 2>&1 || {
    printf 'scontrol is required to validate the NM5 partition time limit\n' >&2
    return 2
  }
  info="$(scontrol show partition "$partition" -o 2>/dev/null)" || {
    printf 'scontrol could not read the selected NM5 partition time limit\n' >&2
    return 2
  }
  max_time="$(awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^MaxTime=/) { sub(/^MaxTime=/, "", $i); print $i; exit } }' <<<"$info")"
  [[ -n "$max_time" ]] || {
    printf 'scontrol response lacks the NM5 partition MaxTime\n' >&2
    return 2
  }
  if [[ "$max_time" != UNLIMITED && "$max_time" != INFINITE && "$max_time" != NOT_SET ]]; then
    max_seconds="$(slurm_duration_seconds "$max_time")" || {
      printf 'scontrol returned an invalid NM5 partition MaxTime\n' >&2
      return 2
    }
    (( requested_seconds <= max_seconds )) || {
      printf 'selected Hydra timeout_seconds exceeds the NM5 partition MaxTime\n' >&2
      return 2
    }
  fi
  days=$((requested_seconds / 86400))
  remaining=$((requested_seconds % 86400))
  hours=$((remaining / 3600))
  minutes=$(((remaining % 3600) / 60))
  seconds=$((remaining % 60))
  if (( days > 0 )); then
    printf -v SUE_SLURM_REQUEST_TIME '%d-%02d:%02d:%02d' "$days" "$hours" "$minutes" "$seconds"
  else
    printf -v SUE_SLURM_REQUEST_TIME '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
  fi
}

validate_hydra_overrides() {
  local stage=$1
  shift
  local override key value
  declare -A seen=()

  for override in "$@"; do
    if [[ "$override" == *$'\n'* || "$override" == *$'\r'* ]]; then
      printf 'newline characters are not permitted in Hydra overrides\n' >&2
      return 2
    fi
    if [[ "$override" != *=* ]]; then
      printf 'Hydra overrides must use KEY=value syntax\n' >&2
      return 2
    fi
    key=${override%%=*}
    value=${override#*=}
    if [[ -z "$key" || -z "$value" || ! "$key" =~ ^[A-Za-z][A-Za-z0-9_.-]*$ ]]; then
      printf 'malformed Hydra override key or value\n' >&2
      return 2
    fi

    case "$key" in
      backend|stage|run_id)
        printf 'override is not permitted for launcher-owned key: %s\n' "$key" >&2
        return 2
        ;;
      assets.*)
        printf 'assets.* overrides are private; set SUE_ASSET_ROOT in backend config\n' >&2
        return 2
        ;;
    esac

    case "$key" in
      method|tracking|scale)
        [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
          printf 'invalid value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        ;;
      seed|train.seed|infer.seed)
        _is_nonnegative_integer "$value" || {
          printf 'invalid value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        if [[ "$stage" == train && ( "$key" == seed || "$key" == train.seed ) && "$value" =~ ^[+]?0+$ ]]; then
          printf 'training seed must be non-zero\n' >&2
          return 2
        fi
        if [[ "$key" == train.seed && "$stage" != train ]]; then
          printf 'training seed override is only valid for the train stage\n' >&2
          return 2
        fi
        ;;
      train.max_steps|train.log_iters|train.timeout_seconds|train.checkpoint_interval_seconds)
        [[ "$stage" == train ]] && _is_positive_integer "$value" || {
          printf 'invalid stage or value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        ;;
      infer.timeout_seconds)
        [[ "$stage" == infer ]] && _is_positive_integer "$value" || {
          printf 'invalid stage or value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        ;;
      train.resume_checkpoint|infer.checkpoint)
        if [[ "$key" == train.resume_checkpoint && "$stage" != train ]] || \
          [[ "$key" == infer.checkpoint && "$stage" != infer ]] || \
          ! _is_bundle_relative_path "$value"; then
          printf 'checkpoint override must be a relative SUE_EXP_DIR path for this stage: %s\n' "$key" >&2
          return 2
        fi
        ;;
      infer.num_samples)
        [[ "$stage" == infer ]] && _is_positive_integer "$value" || {
          printf 'invalid stage or value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        ;;
      infer.num_output_frames)
        [[ "$stage" == infer && "$value" == 123 ]] || {
          printf 'infer.num_output_frames is fixed at 123 latent frames for this profile\n' >&2
          return 2
        }
        ;;
      infer.use_ema)
        [[ "$stage" == infer && ( "$value" == true || "$value" == false ) ]] || {
          printf 'invalid stage or value for Hydra override key: %s\n' "$key" >&2
          return 2
        }
        ;;
      *)
        printf 'unsupported Hydra override key: %s\n' "$key" >&2
        return 2
        ;;
    esac

    if [[ -v seen[$key] ]]; then
      printf 'duplicate Hydra override key: %s\n' "$key" >&2
      return 2
    fi
    seen[$key]=1
  done
}

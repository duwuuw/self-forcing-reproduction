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

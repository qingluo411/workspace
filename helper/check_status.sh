#!/usr/bin/env bash
#
# 汇总 /root/workspace 下各代码仓库的 Git 状态。
# 默认会 fetch 远端（只更新 remote-tracking refs，不改工作区）。

set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORKSPACE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)

KNOWN_REPOS=(LLVM DLCsim DLCSynapse DLC_Custom_Kernel DLC_CL pytorch vllm vllm-cl)
FETCH_REMOTE=1
FETCH_TIMEOUT=20
VERBOSE=0
if [[ -z ${REPORT_TZ:-} ]]; then
    if [[ -e /usr/share/zoneinfo/Asia/Shanghai || -e /usr/lib/zoneinfo/Asia/Shanghai ]]; then
        REPORT_TZ='Asia/Shanghai'
    else
        REPORT_TZ='CST-8'
    fi
fi

usage() {
    cat <<EOF
用法: $(basename "$0") [选项]

检查 $WORKSPACE_ROOT 下各 Git 仓库的分支、工作区、上游跟踪和远端主分支状态。

选项:
  -h, --help          显示帮助
  -o, --offline       不执行 fetch，仅使用本地缓存的远端引用
      --no-fetch      同 --offline
      --timeout N     单个仓库 fetch 超时秒数（默认: $FETCH_TIMEOUT）
  -v, --verbose       显示未提交文件明细（最多 20 条）

环境变量:
  REPORT_TZ           报告时区（默认: Asia/Shanghai；缺少 tzdata 时回退 CST-8）
  NO_COLOR            设置后禁用彩色输出

退出码:
  0  检查完成（即使仓库有未提交改动或落后远端）
  1  未发现 Git 仓库
  2  参数错误
EOF
}

die() {
    printf '错误: %s\n' "$*" >&2
    exit 2
}

while (($# > 0)); do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -o|--offline|--no-fetch)
            FETCH_REMOTE=0
            ;;
        --timeout)
            (($# >= 2)) || die "--timeout 需要一个秒数"
            FETCH_TIMEOUT=$2
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            ;;
        *)
            die "未知参数: $1（使用 --help 查看帮助）"
            ;;
    esac
    shift
done

[[ $FETCH_TIMEOUT =~ ^[1-9][0-9]*$ ]] || die "--timeout 必须是正整数"

if [[ -t 1 && -z ${NO_COLOR:-} ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
else
    C_RESET=''
    C_BOLD=''
    C_DIM=''
    C_RED=''
    C_GREEN=''
    C_YELLOW=''
fi

declare -a REPOS=()
declare -a MISSING=()
declare -A SEEN_REPOS=()

add_repo() {
    local path=$1
    [[ -d $path && -e $path/.git ]] || return 0
    [[ -z ${SEEN_REPOS[$path]:-} ]] || return 0
    SEEN_REPOS[$path]=1
    REPOS+=("$path")
}

for name in "${KNOWN_REPOS[@]}"; do
    if [[ -d $WORKSPACE_ROOT/$name && -e $WORKSPACE_ROOT/$name/.git ]]; then
        add_repo "$WORKSPACE_ROOT/$name"
    else
        MISSING+=("$name")
    fi
done

# 兼容以后新增的仓库：将未列入 KNOWN_REPOS 的直属 Git 仓库追加到末尾。
while IFS= read -r -d '' git_path; do
    add_repo "${git_path%/.git}"
done < <(find "$WORKSPACE_ROOT" -mindepth 2 -maxdepth 2 -name .git -print0 | sort -z)

if ((${#REPOS[@]} == 0)); then
    printf '错误: %s 下未发现 Git 仓库\n' "$WORKSPACE_ROOT" >&2
    exit 1
fi

relation_text() {
    local ahead=$1 behind=$2
    if ((ahead == 0 && behind == 0)); then
        printf '一致'
    elif ((ahead > 0 && behind == 0)); then
        printf '领先 %d' "$ahead"
    elif ((ahead == 0 && behind > 0)); then
        printf '落后 %d' "$behind"
    else
        printf '分叉（领先 %d，落后 %d）' "$ahead" "$behind"
    fi
}

get_relation() {
    local repo=$1 range=$2 ahead behind
    if read -r ahead behind < <(git -C "$repo" rev-list --left-right --count "$range" 2>/dev/null); then
        printf '%s %s\n' "$ahead" "$behind"
        return 0
    fi
    return 1
}

detect_default_branch() {
    local repo=$1 remote=$2 branch

    if branch=$(git -C "$repo" symbolic-ref --quiet --short "refs/remotes/$remote/HEAD" 2>/dev/null); then
        printf '%s\n' "${branch#*/}"
        return 0
    fi

    for branch in main master; do
        if git -C "$repo" show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
            printf '%s\n' "$branch"
            return 0
        fi
    done

    return 1
}

print_relation_colored() {
    local ahead=$1 behind=$2 behind_hint=${3:-} ahead_hint=${4:-}
    local suffix=''
    if ((ahead == 0 && behind == 0)); then
        printf '%b一致%b' "$C_GREEN" "$C_RESET"
    elif ((behind > 0)); then
        [[ -z $behind_hint ]] || suffix="（$behind_hint）"
        printf '%b%s%s%b' "$C_RED" "$(relation_text "$ahead" "$behind")" "$suffix" "$C_RESET"
    else
        [[ -z $ahead_hint ]] || suffix="（$ahead_hint）"
        printf '%b%s%s%b' "$C_YELLOW" "$(relation_text "$ahead" "$behind")" "$suffix" "$C_RESET"
    fi
}

report_time=$(TZ="$REPORT_TZ" date '+%Y-%m-%d %H:%M:%S %Z')
if ((FETCH_REMOTE)); then
    fetch_mode='开启'
else
    fetch_mode='关闭（使用本地缓存）'
fi

printf '%bDLC 软件栈仓库状态%b — %s\n' "$C_BOLD" "$C_RESET" "$report_time"
printf '%b工作区%b: %s\n' "$C_DIM" "$C_RESET" "$WORKSPACE_ROOT"
printf '%b远端探测%b: fetch %s' "$C_DIM" "$C_RESET" "$fetch_mode"
if ((FETCH_REMOTE)); then
    printf '，单仓库超时 %ss' "$FETCH_TIMEOUT"
fi
printf '\n'

clean_count=0
dirty_count=0
fetch_failed_count=0
main_behind_count=0
no_upstream_count=0
repo_index=0

for repo in "${REPOS[@]}"; do
    ((repo_index += 1))
    repo_name=${repo#"$WORKSPACE_ROOT"/}

    remote=''
    if git -C "$repo" remote get-url origin >/dev/null 2>&1; then
        remote='origin'
    else
        remote=$(git -C "$repo" remote 2>/dev/null | sed -n '1p')
    fi

    default_branch=''
    if [[ -n $remote ]]; then
        default_branch=$(detect_default_branch "$repo" "$remote" 2>/dev/null || true)
    fi
    old_remote_tip=''
    if [[ -n $remote && -n $default_branch ]]; then
        old_remote_tip=$(git -C "$repo" rev-parse --verify --quiet "refs/remotes/$remote/$default_branch" 2>/dev/null || true)
    fi

    fetch_ok=1
    fetch_error=''
    if ((FETCH_REMOTE)) && [[ -n $remote ]]; then
        if ! fetch_error=$(GIT_TERMINAL_PROMPT=0 timeout "$FETCH_TIMEOUT" git -C "$repo" fetch --quiet --prune --no-tags "$remote" 2>&1); then
            fetch_ok=0
            ((fetch_failed_count += 1))
            fetch_error=${fetch_error//$'\n'/; }
            fetch_error=${fetch_error:0:180}
            [[ -n $fetch_error ]] || fetch_error="超时或未知错误"
        fi
    elif [[ -z $remote ]]; then
        fetch_ok=0
        fetch_error='未配置 remote'
    fi

    if [[ -n $remote && -z $default_branch ]]; then
        default_branch=$(detect_default_branch "$repo" "$remote" 2>/dev/null || true)
    fi
    remote_ref=''
    new_remote_tip=''
    remote_update_note=''
    if [[ -n $remote && -n $default_branch ]]; then
        remote_ref="refs/remotes/$remote/$default_branch"
        new_remote_tip=$(git -C "$repo" rev-parse --verify --quiet "$remote_ref" 2>/dev/null || true)
    fi

    if ((!FETCH_REMOTE)); then
        remote_update_note='未联网，显示本地缓存'
    elif ((fetch_ok == 0)) && [[ -n $remote ]]; then
        remote_update_note="fetch 失败"
    elif [[ -z $remote ]]; then
        remote_update_note='未配置 remote'
    elif [[ -z $new_remote_tip ]]; then
        remote_update_note='未找到远端主分支'
    elif [[ -z $old_remote_tip ]]; then
        remote_update_note='首次获取远端分支'
    elif [[ $old_remote_tip != "$new_remote_tip" ]]; then
        remote_update_note="本轮获取到新提交 ${old_remote_tip:0:9} -> ${new_remote_tip:0:9}"
    else
        remote_update_note='本轮无新提交'
    fi

    if branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null); then
        branch_display=$branch
        detached=0
    else
        branch_display="detached@$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
        detached=1
    fi

    upstream=$(git -C "$repo" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
    upstream_ahead=''
    upstream_behind=''
    if [[ -n $upstream ]]; then
        if read -r upstream_ahead upstream_behind < <(get_relation "$repo" "HEAD...$upstream"); then
            :
        else
            upstream=''
        fi
    fi
    if [[ -z $upstream ]]; then
        ((no_upstream_count += 1))
    fi

    head_info=$(git -C "$repo" log -1 --format='%h %cs %s' 2>/dev/null || true)
    [[ -n $head_info ]] || head_info='无可读取提交'

    status_raw=$(git -C "$repo" status --porcelain=v1 --untracked-files=normal 2>/dev/null || true)
    total_changes=$(printf '%s\n' "$status_raw" | awk 'NF {n++} END {print n+0}')
    tracked_changes=$(printf '%s\n' "$status_raw" | awk 'NF && substr($0, 1, 2) != "??" {n++} END {print n+0}')
    untracked_changes=$(printf '%s\n' "$status_raw" | awk 'NF && substr($0, 1, 2) == "??" {n++} END {print n+0}')
    staged_changes=$(git -C "$repo" diff --cached --name-only 2>/dev/null | awk 'NF {n++} END {print n+0}')
    unstaged_changes=$(git -C "$repo" diff --name-only 2>/dev/null | awk 'NF {n++} END {print n+0}')
    conflict_changes=$(git -C "$repo" diff --name-only --diff-filter=U 2>/dev/null | awk 'NF {n++} END {print n+0}')
    stash_count=$(git -C "$repo" stash list 2>/dev/null | awk 'NF {n++} END {print n+0}')

    if ((total_changes == 0)); then
        ((clean_count += 1))
    else
        ((dirty_count += 1))
    fi

    local_main_behind=0
    local_main_ahead=0
    if [[ -n $remote && -n $default_branch ]]; then
        if git -C "$repo" show-ref --verify --quiet "refs/heads/$default_branch"; then
            if read -r local_main_ahead local_main_behind < <(get_relation "$repo" "refs/heads/$default_branch...$remote_ref"); then
                if ((local_main_behind > 0)); then
                    ((main_behind_count += 1))
                fi
            fi
        fi
    fi

    if ((total_changes == 0)); then
        status_tag="${C_GREEN}[干净]${C_RESET}"
    else
        status_tag="${C_YELLOW}[有改动]${C_RESET}"
    fi
    if ((fetch_ok == 0)); then
        status_tag+=" ${C_RED}[远端不可用]${C_RESET}"
    fi

    printf '\n%b[%d/%d] %s%b %s\n' "$C_BOLD" "$repo_index" "${#REPOS[@]}" "$repo_name" "$C_RESET" "$status_tag"

    if ((detached)); then
        printf '  分支: %b%s%b\n' "$C_YELLOW" "$branch_display" "$C_RESET"
    elif [[ -n $upstream ]]; then
        printf '  分支: %s  ->  %s  [%b%s%b]\n' \
            "$branch_display" "$upstream" \
            "$( ((upstream_ahead == 0 && upstream_behind == 0)) && printf '%s' "$C_GREEN" || printf '%s' "$C_YELLOW")" \
            "$(relation_text "$upstream_ahead" "$upstream_behind")" "$C_RESET"
    else
        printf '  分支: %s  %b[无上游分支]%b\n' "$branch_display" "$C_YELLOW" "$C_RESET"
    fi
    printf '  HEAD: %s\n' "$head_info"

    local_branches=()
    while IFS= read -r local_branch; do
        [[ -n $local_branch ]] || continue
        if ((!detached)) && [[ $local_branch == "$branch_display" ]]; then
            local_branches+=("${C_GREEN}* ${local_branch}${C_RESET}")
        else
            local_branches+=("$local_branch")
        fi
    done < <(git -C "$repo" for-each-ref --format='%(refname:short)' refs/heads 2>/dev/null | LC_ALL=C sort)
    if ((${#local_branches[@]} > 0)); then
        printf '  本地分支 (%d): %s\n' "${#local_branches[@]}" "${local_branches[*]}"
    else
        printf '  本地分支: %b无%b\n' "$C_DIM" "$C_RESET"
    fi

    if [[ -n $remote && -n $default_branch && -n $new_remote_tip ]]; then
        remote_info=$(git -C "$repo" log -1 --format='%h %cs %s' "$remote_ref" 2>/dev/null || true)
        if [[ $remote_update_note == '本轮获取到新提交'* ]]; then
            note_color=$C_YELLOW
        elif [[ $remote_update_note == '本轮无新提交' ]]; then
            note_color=$C_GREEN
        else
            note_color=$C_DIM
        fi
        printf '  远端主分支: %s/%s @ %s  [%b%s%b]\n' \
            "$remote" "$default_branch" "$remote_info" "$note_color" "$remote_update_note" "$C_RESET"

        if git -C "$repo" show-ref --verify --quiet "refs/heads/$default_branch"; then
            printf '  本地 %s 相对 %s/%s: ' "$default_branch" "$remote" "$default_branch"
            print_relation_colored "$local_main_ahead" "$local_main_behind" \
                '主分支需更新' '本地主分支有未推送提交'
            printf '\n'
        else
            printf '  本地 %s: %b不存在%b\n' "$default_branch" "$C_YELLOW" "$C_RESET"
        fi

        if [[ $branch_display != "$default_branch" ]]; then
            current_ahead=''
            current_behind=''
            if read -r current_ahead current_behind < <(get_relation "$repo" "HEAD...$remote_ref"); then
                printf '  当前 HEAD 相对 %s/%s: ' "$remote" "$default_branch"
                print_relation_colored "$current_ahead" "$current_behind" \
                    '远端主分支有更新' ''
                printf '\n'
            fi
        fi
    else
        printf '  远端主分支: %b%s%b\n' "$C_DIM" "$remote_update_note" "$C_RESET"
    fi

    if ((total_changes == 0)); then
        printf '  工作区: %b干净%b\n' "$C_GREEN" "$C_RESET"
    else
        printf '  工作区: %b%d 处改动%b（跟踪文件 %d：已暂存 %d，未暂存 %d；未跟踪 %d）\n' \
            "$C_YELLOW" "$total_changes" "$C_RESET" "$tracked_changes" \
            "$staged_changes" "$unstaged_changes" "$untracked_changes"
    fi
    if ((conflict_changes > 0)); then
        printf '  冲突: %b%d 个文件%b\n' "$C_RED" "$conflict_changes" "$C_RESET"
    fi
    if ((stash_count > 0)); then
        printf '  Stash: %d 条\n' "$stash_count"
    fi

    if [[ -f $repo/.gitmodules ]]; then
        submodule_total=$(git -C "$repo" config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | awk 'NF {n++} END {print n+0}')
        if ((submodule_total > 0)); then
            read -r submodule_recursive_total submodule_bad < <(
                git -C "$repo" submodule status --recursive 2>/dev/null |
                    awk 'NF {n++; if (substr($0, 1, 1) ~ /[+-U]/) bad++} END {print n+0, bad+0}'
            )
            if ((submodule_bad == 0)); then
                printf '  子模块: 顶层 %d 个，递归 %d 个，状态正常\n' \
                    "$submodule_total" "$submodule_recursive_total"
            else
                printf '  子模块: 顶层 %d 个，递归 %d 个，%b%d 个未初始化或版本不一致%b\n' \
                    "$submodule_total" "$submodule_recursive_total" \
                    "$C_YELLOW" "$submodule_bad" "$C_RESET"
            fi
        fi
    fi

    if ((fetch_ok == 0)) && [[ -n $fetch_error ]]; then
        printf '  远端错误: %b%s%b\n' "$C_RED" "$fetch_error" "$C_RESET"
    fi

    if ((VERBOSE && total_changes > 0)); then
        printf '  改动明细（最多 20 条）:\n'
        printf '%s\n' "$status_raw" | awk 'NF {n++; if (n <= 20) print "    " $0} END {if (n > 20) print "    ... 另有 " (n - 20) " 条"}'
    fi
done

printf '\n%b汇总%b: %d 个仓库 | 工作区干净 %b%d%b | 有改动 %b%d%b' \
    "$C_BOLD" "$C_RESET" "${#REPOS[@]}" \
    "$C_GREEN" "$clean_count" "$C_RESET" \
    "$C_YELLOW" "$dirty_count" "$C_RESET"
if ((FETCH_REMOTE)); then
    printf ' | fetch 失败 %b%d%b' "$C_RED" "$fetch_failed_count" "$C_RESET"
fi
printf ' | 本地主分支落后 %b%d%b | 无上游 %b%d%b\n' \
    "$C_RED" "$main_behind_count" "$C_RESET" \
    "$C_YELLOW" "$no_upstream_count" "$C_RESET"

if ((${#MISSING[@]} > 0)); then
    printf '缺少预期仓库: %b%s%b\n' "$C_RED" "${MISSING[*]}" "$C_RESET"
fi

exit 0

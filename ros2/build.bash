#!/usr/bin/env bash
set -e
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$workspace/env.bash" --base
cd "$workspace"
arguments=("$@")
if [[ "$ALOHAMINI_ROS_DISTRO" == humble && -d vendor/humble ]]; then
    # Keep the patched TF core in every Humble build, including custom package
    # selections and simulation builds that supply their own source paths.
    found_paths=false
    for ((index=0; index<${#arguments[@]}; index++)); do
        if [[ "${arguments[index]}" == --base-paths ]]; then
            arguments=("${arguments[@]:0:index+1}" vendor/humble "${arguments[@]:index+1}")
            found_paths=true
            break
        fi
    done
    if [[ "$found_paths" == false ]]; then
        arguments=(--base-paths src vendor/humble "${arguments[@]}")
    fi
fi
exec "$ALOHAMINI_ROS_PYTHON" -m colcon --log-base "log/$ALOHAMINI_ROS_DISTRO" build \
    --symlink-install \
    --build-base "build/$ALOHAMINI_ROS_DISTRO" \
    --install-base "install/$ALOHAMINI_ROS_DISTRO" \
    --cmake-args "-DPython3_EXECUTABLE=$ALOHAMINI_ROS_PYTHON" "${arguments[@]}"

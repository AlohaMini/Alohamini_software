#!/usr/bin/env bash
set -e
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$workspace/env.bash" --base
cd "$workspace"
arguments=("$@")
source_paths=(src)
if [[ "$ALOHAMINI_ROS_DISTRO" == humble && -d vendor/humble ]]; then
    source_paths+=(vendor/humble)
fi
found_paths=false
for ((index=0; index<${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == --base-paths ]]; then
        # Include the patched TF core in custom Humble builds as well.
        if [[ "$ALOHAMINI_ROS_DISTRO" == humble && -d vendor/humble ]]; then
            arguments=("${arguments[@]:0:index+1}" vendor/humble "${arguments[@]:index+1}")
        fi
        found_paths=true
        break
    fi
done
if [[ "$found_paths" == false ]]; then
    # Do not let default recursive discovery pick another distro's vendors.
    arguments=(--base-paths "${source_paths[@]}" "${arguments[@]}")
fi
exec "$ALOHAMINI_ROS_PYTHON" -m colcon --log-base "log/$ALOHAMINI_ROS_DISTRO" build \
    --symlink-install \
    --build-base "build/$ALOHAMINI_ROS_DISTRO" \
    --install-base "install/$ALOHAMINI_ROS_DISTRO" \
    --cmake-args "-DPython3_EXECUTABLE=$ALOHAMINI_ROS_PYTHON" "${arguments[@]}"

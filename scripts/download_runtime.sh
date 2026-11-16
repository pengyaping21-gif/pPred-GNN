#!/usr/bin/env bash
# Download and verify a complete torch1 environment or Docker image.
set -euo pipefail

kind=${1:-docker}
destination=${2:-.cache/runtime}
case "$kind" in
  docker) archive=ppred-torch1-linux-amd64.docker.tar ;;
  conda) archive=torch1-conda-linux-amd64.tar.gz ;;
  *) printf 'Usage: bash scripts/download_runtime.sh [docker|conda] [directory]\n' >&2; exit 2 ;;
esac
release=https://github.com/pengyaping21-gif/pPred-GNN/releases/download/torch1-runtime-20261116
mkdir -p "$destination"
cd "$destination"
curl -fL --retry 3 "$release/$archive.parts.sha256" -o "$archive.parts.sha256"
curl -fL --retry 3 "$release/$archive.sha256" -o "$archive.sha256"
read -r archive_checksum archive_name < "$archive.sha256"
[[ "$archive_checksum" =~ ^[0-9a-f]{64}$ && "$archive_name" == "$archive" ]] || {
  printf 'Invalid archive checksum\n' >&2; exit 1;
}
while read -r checksum part; do
  [[ "$checksum" =~ ^[0-9a-f]{64}$ && "$part" == "$archive".part-[0-9][0-9][0-9] ]] || {
    printf 'Invalid part manifest\n' >&2; exit 1;
  }
  if [[ ! -f "$part" ]] || ! printf '%s  %s\n' "$checksum" "$part" | sha256sum -c --status; then
    curl -fL --retry 3 "$release/$part" -o "$part.download"
    mv "$part.download" "$part"
  fi
done < "$archive.parts.sha256"
sha256sum -c "$archive.parts.sha256"
mapfile -t parts < <(awk '{print $2}' "$archive.parts.sha256")
(( ${#parts[@]} > 0 )) || { printf 'Empty part manifest\n' >&2; exit 1; }
cat "${parts[@]}" > "$archive.download"
printf '%s  %s\n' "$archive_checksum" "$archive.download" | sha256sum -c -
mv "$archive.download" "$archive"
printf '\nReady: %s/%s\n' "$PWD" "$archive"

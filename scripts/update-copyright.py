# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os

default_header = """Copyright (c) 2018-2025 by xcube team and contributors
Permissions are hereby granted under the terms of the MIT License:
https://opensource.org/licenses/MIT."""


def update_copyright(
    dir_path: str = ".",
    header: str | None = None,
    prefix: str = "#",
    ext: str = ".py",
):
    header_lines = list(
        map(lambda line: f"{prefix} {line}\n", (header or default_header).split("\n"))
    )
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(ext) and filename != "update-copyright.py":
                _update_copyright(os.path.join(root, filename), header_lines, prefix)


def _update_copyright(file_path: str, header_lines: list[str], prefix: str):
    with open(file_path) as f:
        lines = f.readlines()

    old_header_size = 0
    is_copyright = False
    for line in lines:
        if line.startswith(prefix):
            is_copyright = is_copyright or "copyright" in line.lower()
            old_header_size += 1
        else:
            break

    if is_copyright or old_header_size == 0:
        with open(file_path, "w") as f:
            f.writelines(header_lines)
            if old_header_size == 0:
                f.write("\n")
            f.writelines(lines[old_header_size:])
        if old_header_size == 0:
            print("Added missing CR header: ", file_path)
    else:
        print("Not a CR header: ", file_path)


update_copyright("../xcube")
update_copyright("../test")

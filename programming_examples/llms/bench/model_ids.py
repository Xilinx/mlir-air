# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Associate flat or nested LLM examples' LIT verification with profile IDs."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


def verification_status(xml_file, llms_root="programming_examples/llms"):
    aliases = {}
    root = Path(llms_root)
    for lit in root.rglob("run_npu2_profile*.lit"):
        directory = lit.parent.relative_to(root).as_posix()
        text = lit.read_text()
        names = re.findall(r"^//\s*PERF_MODEL:\s*(\S+)", text, re.M)
        names += re.findall(r"extract_perf\.py\b.*--model\s+(\S+)", text)
        aliases.setdefault(directory, set()).update(names)
    result = {}
    rank = {"skip": 0, "pass": 1, "fail": 2}
    if not Path(xml_file).exists():
        return result
    for case in ET.parse(xml_file).getroot().iter("testcase"):
        name = (case.get("classname") or "") + " " + (case.get("name") or "")
        match = re.search(r"llms/([^ ]+?)[ /]run_npu2_verify", name)
        if not match:
            continue
        directory = match.group(1)
        status = (
            "fail"
            if next(case.iter("failure"), None) is not None
            else "skip" if next(case.iter("skipped"), None) is not None else "pass"
        )
        names = aliases.get(directory) or {directory.replace("/", "_")}
        for model in names:
            if model not in result or rank[status] > rank[result[model]]:
                result[model] = status
    return result

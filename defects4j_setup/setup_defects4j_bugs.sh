#!/bin/bash
# Set up a batch of Defects4J bugs for HDD-LOC: checks out BOTH the buggy
# and fixed version of each bug (you need both for the differential
# oracle), compiles both, and pulls out everything you need to write the
# per-bug BugAdapter and point defects4j_oracle.py at it.
#
# Fixes vs. the earlier version of this script:
#   - "JXPath" -> "JxPath" (project IDs are case-sensitive; JxPath is the
#     actually-registered one)
#   - removed `defects4j diff`, which does not exist as a command -- the
#     ground-truth patch is read from the file Defects4J already ships at
#     framework/projects/<Proj>/patches/<bid>.src.patch instead
#   - actually checks out + compiles both versions (was commented out)
#   - exports the trigger test name and each checkout's compiled-classes
#     path, since you need both to write BugAdapter.java and to point
#     JavaHarnessClient(classpath=...) at the right directory

set -u

declare -A BUGS=(
  ["JacksonXml"]="1 2 3 4 5"
  ["JacksonCore"]="1 3 4 5"
  ["JacksonDatabind"]="1 3 5 8"
  ["Gson"]="1 2 6"
  ["Jsoup"]="1 2 27"
  ["JxPath"]="1"
)

D4J_HOME="$(cd "$(dirname "$(which defects4j)")/../.." && pwd)"
OUT_DIR="$HOME/Desktop/research/structured_bugs"
mkdir -p "$OUT_DIR"

for proj in "${!BUGS[@]}"; do
  for bid in ${BUGS[$proj]}; do
    echo "=== $proj-$bid ==="
    bug_dir="$OUT_DIR/${proj}_${bid}"
    buggy_dir="${bug_dir}_buggy"
    fixed_dir="${bug_dir}_fixed"
    mkdir -p "$bug_dir"

    # 1. Checkout both versions (b = buggy, f = fixed) -- you need both
    #    compiled for the differential oracle (buggy vs golden output).
    defects4j checkout -p "$proj" -v "${bid}b" -w "$buggy_dir"
    defects4j checkout -p "$proj" -v "${bid}f" -w "$fixed_dir"

    # 2. Compile both.
    defects4j compile -w "$buggy_dir"
    defects4j compile -w "$fixed_dir"

    # 3. Export the compiled-classes path for each -- this is exactly what
    #    goes into JavaHarnessClient(classpath=...) later. Different
    #    projects use different build layouts, so don't hardcode
    #    "target/classes" -- ask Defects4J for the real path.
    defects4j export -p dir.bin.classes -w "$buggy_dir" -o "$bug_dir/buggy_classes_path.txt"
    defects4j export -p dir.bin.classes -w "$fixed_dir" -o "$bug_dir/fixed_classes_path.txt"

    # 4. Export the trigger test(s) -- read these to figure out exactly
    #    what data/input the bug's trigger test constructs and passes into
    #    the buggy method. That's the "input" you'll represent as a Python
    #    structure and feed to hdd_loc.py.
    defects4j export -p tests.trigger -w "$buggy_dir" -o "$bug_dir/trigger_tests.txt"

    # 5. Copy the ground-truth patch Defects4J already ships -- this tells
    #    you the actual fixed line(s), which is what you check your SBFL
    #    ranking against later.
    patch_file="$D4J_HOME/framework/projects/$proj/patches/${bid}.src.patch"
    if [ -f "$patch_file" ]; then
        cp "$patch_file" "$bug_dir/ground_truth.patch"
    else
        echo "  (warning: no patch file found at $patch_file)"
    fi

    echo "  trigger test(s): $(cat "$bug_dir/trigger_tests.txt" 2>/dev/null | tr '\n' ' ')"
    echo "  buggy classes:   $(cat "$bug_dir/buggy_classes_path.txt" 2>/dev/null)"
    echo "  fixed classes:   $(cat "$bug_dir/fixed_classes_path.txt" 2>/dev/null)"
    echo
  done
done

echo "Done. Each bug's checkout, classpaths, trigger test name, and ground-truth patch are under $OUT_DIR/<proj>_<bid>/"
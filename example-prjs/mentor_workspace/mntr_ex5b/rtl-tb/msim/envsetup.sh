# Usage:
# $ source envsetup.sh
#

# Base directory for CAD tools.
export CAD_PATH="/opt/cad"

# We do need Mentor Modelsim (simulator).
export PATH="${CAD_PATH}/msim/modeltech/bin":$PATH

# Mentor Graphics
export LM_LICENSE_FILE=${LM_LICENSE_FILE}:"1720@bioeecad.ee.columbia.edu"

export PS1="[modelsim] $PS1"

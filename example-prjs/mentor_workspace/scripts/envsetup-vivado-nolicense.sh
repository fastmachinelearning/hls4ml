# Usage:
# $ source envsetup.sh
#

# Base directory for CAD tools.
export CAD_PATH=/opt/cad

# We do not need licensing for this example.
#export XILINXD_LICENSE_FILE="2177@espdev.cs.columbia.edu"

source $CAD_PATH/vivado/settings64.sh

export PS1="[vivado-nolicense] $PS1"

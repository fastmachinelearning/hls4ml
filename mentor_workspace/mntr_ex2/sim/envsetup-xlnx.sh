#
# Export few environment variables to compile the example with Xilinx's
# provided libraries.
#
# Usage:
# $ source envsetup-xlnx.sh
#

# Base directory for CAD tools.
export CAD_PATH=/opt/cad

# We do not need licensing for this example.

# Let's source the usual Vivado script.
source ${CAD_PATH}/vivado/settings64.sh

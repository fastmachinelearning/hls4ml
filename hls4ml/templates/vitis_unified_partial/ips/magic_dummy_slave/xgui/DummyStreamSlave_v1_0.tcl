# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SINK_MODE" -parent ${Page_0}


}

proc update_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to update DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to validate DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.SINK_MODE { PARAM_VALUE.SINK_MODE } {
	# Procedure called to update SINK_MODE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SINK_MODE { PARAM_VALUE.SINK_MODE } {
	# Procedure called to validate SINK_MODE
	return true
}

proc update_PARAM_VALUE.STORAGE_IDX_WIDTH { PARAM_VALUE.STORAGE_IDX_WIDTH } {
	# Procedure called to update STORAGE_IDX_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STORAGE_IDX_WIDTH { PARAM_VALUE.STORAGE_IDX_WIDTH } {
	# Procedure called to validate STORAGE_IDX_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.DATA_WIDTH { MODELPARAM_VALUE.DATA_WIDTH PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DATA_WIDTH}] ${MODELPARAM_VALUE.DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.STORAGE_IDX_WIDTH { MODELPARAM_VALUE.STORAGE_IDX_WIDTH PARAM_VALUE.STORAGE_IDX_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STORAGE_IDX_WIDTH}] ${MODELPARAM_VALUE.STORAGE_IDX_WIDTH}
}

proc update_MODELPARAM_VALUE.SINK_MODE { MODELPARAM_VALUE.SINK_MODE PARAM_VALUE.SINK_MODE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SINK_MODE}] ${MODELPARAM_VALUE.SINK_MODE}
}


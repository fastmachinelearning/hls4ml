# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK0_CNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK0_CONTROL_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK0_INTR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK0_ROUNDTRIP_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK0_STATUS_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_DST_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_DST_SIZE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_INDEX_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_LD_MSK_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_PROFILE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_SRC_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_SRC_SIZE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_STATUS_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BANK1_ST_MSK_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DMA_EXEC_TASK_CNT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DMA_INIT_TASK_CNT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GLOB_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "GLOB_DATA_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.ADDR_WIDTH { PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to update ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ADDR_WIDTH { PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to validate ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK0_CNT_WIDTH { PARAM_VALUE.BANK0_CNT_WIDTH } {
	# Procedure called to update BANK0_CNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK0_CNT_WIDTH { PARAM_VALUE.BANK0_CNT_WIDTH } {
	# Procedure called to validate BANK0_CNT_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK0_CONTROL_WIDTH { PARAM_VALUE.BANK0_CONTROL_WIDTH } {
	# Procedure called to update BANK0_CONTROL_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK0_CONTROL_WIDTH { PARAM_VALUE.BANK0_CONTROL_WIDTH } {
	# Procedure called to validate BANK0_CONTROL_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK0_INTR_WIDTH { PARAM_VALUE.BANK0_INTR_WIDTH } {
	# Procedure called to update BANK0_INTR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK0_INTR_WIDTH { PARAM_VALUE.BANK0_INTR_WIDTH } {
	# Procedure called to validate BANK0_INTR_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH { PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH } {
	# Procedure called to update BANK0_ROUNDTRIP_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH { PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH } {
	# Procedure called to validate BANK0_ROUNDTRIP_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK0_STATUS_WIDTH { PARAM_VALUE.BANK0_STATUS_WIDTH } {
	# Procedure called to update BANK0_STATUS_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK0_STATUS_WIDTH { PARAM_VALUE.BANK0_STATUS_WIDTH } {
	# Procedure called to validate BANK0_STATUS_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_DST_ADDR_WIDTH { PARAM_VALUE.BANK1_DST_ADDR_WIDTH } {
	# Procedure called to update BANK1_DST_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_DST_ADDR_WIDTH { PARAM_VALUE.BANK1_DST_ADDR_WIDTH } {
	# Procedure called to validate BANK1_DST_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_DST_SIZE_WIDTH { PARAM_VALUE.BANK1_DST_SIZE_WIDTH } {
	# Procedure called to update BANK1_DST_SIZE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_DST_SIZE_WIDTH { PARAM_VALUE.BANK1_DST_SIZE_WIDTH } {
	# Procedure called to validate BANK1_DST_SIZE_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_INDEX_WIDTH { PARAM_VALUE.BANK1_INDEX_WIDTH } {
	# Procedure called to update BANK1_INDEX_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_INDEX_WIDTH { PARAM_VALUE.BANK1_INDEX_WIDTH } {
	# Procedure called to validate BANK1_INDEX_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_LD_MSK_WIDTH { PARAM_VALUE.BANK1_LD_MSK_WIDTH } {
	# Procedure called to update BANK1_LD_MSK_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_LD_MSK_WIDTH { PARAM_VALUE.BANK1_LD_MSK_WIDTH } {
	# Procedure called to validate BANK1_LD_MSK_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_PROFILE_WIDTH { PARAM_VALUE.BANK1_PROFILE_WIDTH } {
	# Procedure called to update BANK1_PROFILE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_PROFILE_WIDTH { PARAM_VALUE.BANK1_PROFILE_WIDTH } {
	# Procedure called to validate BANK1_PROFILE_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_SRC_ADDR_WIDTH { PARAM_VALUE.BANK1_SRC_ADDR_WIDTH } {
	# Procedure called to update BANK1_SRC_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_SRC_ADDR_WIDTH { PARAM_VALUE.BANK1_SRC_ADDR_WIDTH } {
	# Procedure called to validate BANK1_SRC_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_SRC_SIZE_WIDTH { PARAM_VALUE.BANK1_SRC_SIZE_WIDTH } {
	# Procedure called to update BANK1_SRC_SIZE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_SRC_SIZE_WIDTH { PARAM_VALUE.BANK1_SRC_SIZE_WIDTH } {
	# Procedure called to validate BANK1_SRC_SIZE_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_STATUS_WIDTH { PARAM_VALUE.BANK1_STATUS_WIDTH } {
	# Procedure called to update BANK1_STATUS_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_STATUS_WIDTH { PARAM_VALUE.BANK1_STATUS_WIDTH } {
	# Procedure called to validate BANK1_STATUS_WIDTH
	return true
}

proc update_PARAM_VALUE.BANK1_ST_MSK_WIDTH { PARAM_VALUE.BANK1_ST_MSK_WIDTH } {
	# Procedure called to update BANK1_ST_MSK_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BANK1_ST_MSK_WIDTH { PARAM_VALUE.BANK1_ST_MSK_WIDTH } {
	# Procedure called to validate BANK1_ST_MSK_WIDTH
	return true
}

proc update_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to update DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DATA_WIDTH { PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to validate DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.DMA_EXEC_TASK_CNT { PARAM_VALUE.DMA_EXEC_TASK_CNT } {
	# Procedure called to update DMA_EXEC_TASK_CNT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DMA_EXEC_TASK_CNT { PARAM_VALUE.DMA_EXEC_TASK_CNT } {
	# Procedure called to validate DMA_EXEC_TASK_CNT
	return true
}

proc update_PARAM_VALUE.DMA_INIT_TASK_CNT { PARAM_VALUE.DMA_INIT_TASK_CNT } {
	# Procedure called to update DMA_INIT_TASK_CNT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DMA_INIT_TASK_CNT { PARAM_VALUE.DMA_INIT_TASK_CNT } {
	# Procedure called to validate DMA_INIT_TASK_CNT
	return true
}

proc update_PARAM_VALUE.GLOB_ADDR_WIDTH { PARAM_VALUE.GLOB_ADDR_WIDTH } {
	# Procedure called to update GLOB_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GLOB_ADDR_WIDTH { PARAM_VALUE.GLOB_ADDR_WIDTH } {
	# Procedure called to validate GLOB_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.GLOB_DATA_WIDTH { PARAM_VALUE.GLOB_DATA_WIDTH } {
	# Procedure called to update GLOB_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.GLOB_DATA_WIDTH { PARAM_VALUE.GLOB_DATA_WIDTH } {
	# Procedure called to validate GLOB_DATA_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.GLOB_ADDR_WIDTH { MODELPARAM_VALUE.GLOB_ADDR_WIDTH PARAM_VALUE.GLOB_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GLOB_ADDR_WIDTH}] ${MODELPARAM_VALUE.GLOB_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.GLOB_DATA_WIDTH { MODELPARAM_VALUE.GLOB_DATA_WIDTH PARAM_VALUE.GLOB_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.GLOB_DATA_WIDTH}] ${MODELPARAM_VALUE.GLOB_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.ADDR_WIDTH { MODELPARAM_VALUE.ADDR_WIDTH PARAM_VALUE.ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ADDR_WIDTH}] ${MODELPARAM_VALUE.ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.DATA_WIDTH { MODELPARAM_VALUE.DATA_WIDTH PARAM_VALUE.DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DATA_WIDTH}] ${MODELPARAM_VALUE.DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_INDEX_WIDTH { MODELPARAM_VALUE.BANK1_INDEX_WIDTH PARAM_VALUE.BANK1_INDEX_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_INDEX_WIDTH}] ${MODELPARAM_VALUE.BANK1_INDEX_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_SRC_ADDR_WIDTH { MODELPARAM_VALUE.BANK1_SRC_ADDR_WIDTH PARAM_VALUE.BANK1_SRC_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_SRC_ADDR_WIDTH}] ${MODELPARAM_VALUE.BANK1_SRC_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_SRC_SIZE_WIDTH { MODELPARAM_VALUE.BANK1_SRC_SIZE_WIDTH PARAM_VALUE.BANK1_SRC_SIZE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_SRC_SIZE_WIDTH}] ${MODELPARAM_VALUE.BANK1_SRC_SIZE_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_DST_ADDR_WIDTH { MODELPARAM_VALUE.BANK1_DST_ADDR_WIDTH PARAM_VALUE.BANK1_DST_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_DST_ADDR_WIDTH}] ${MODELPARAM_VALUE.BANK1_DST_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_DST_SIZE_WIDTH { MODELPARAM_VALUE.BANK1_DST_SIZE_WIDTH PARAM_VALUE.BANK1_DST_SIZE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_DST_SIZE_WIDTH}] ${MODELPARAM_VALUE.BANK1_DST_SIZE_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_STATUS_WIDTH { MODELPARAM_VALUE.BANK1_STATUS_WIDTH PARAM_VALUE.BANK1_STATUS_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_STATUS_WIDTH}] ${MODELPARAM_VALUE.BANK1_STATUS_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_PROFILE_WIDTH { MODELPARAM_VALUE.BANK1_PROFILE_WIDTH PARAM_VALUE.BANK1_PROFILE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_PROFILE_WIDTH}] ${MODELPARAM_VALUE.BANK1_PROFILE_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_LD_MSK_WIDTH { MODELPARAM_VALUE.BANK1_LD_MSK_WIDTH PARAM_VALUE.BANK1_LD_MSK_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_LD_MSK_WIDTH}] ${MODELPARAM_VALUE.BANK1_LD_MSK_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK1_ST_MSK_WIDTH { MODELPARAM_VALUE.BANK1_ST_MSK_WIDTH PARAM_VALUE.BANK1_ST_MSK_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK1_ST_MSK_WIDTH}] ${MODELPARAM_VALUE.BANK1_ST_MSK_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK0_CONTROL_WIDTH { MODELPARAM_VALUE.BANK0_CONTROL_WIDTH PARAM_VALUE.BANK0_CONTROL_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK0_CONTROL_WIDTH}] ${MODELPARAM_VALUE.BANK0_CONTROL_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK0_STATUS_WIDTH { MODELPARAM_VALUE.BANK0_STATUS_WIDTH PARAM_VALUE.BANK0_STATUS_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK0_STATUS_WIDTH}] ${MODELPARAM_VALUE.BANK0_STATUS_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK0_CNT_WIDTH { MODELPARAM_VALUE.BANK0_CNT_WIDTH PARAM_VALUE.BANK0_CNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK0_CNT_WIDTH}] ${MODELPARAM_VALUE.BANK0_CNT_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK0_INTR_WIDTH { MODELPARAM_VALUE.BANK0_INTR_WIDTH PARAM_VALUE.BANK0_INTR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK0_INTR_WIDTH}] ${MODELPARAM_VALUE.BANK0_INTR_WIDTH}
}

proc update_MODELPARAM_VALUE.BANK0_ROUNDTRIP_WIDTH { MODELPARAM_VALUE.BANK0_ROUNDTRIP_WIDTH PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BANK0_ROUNDTRIP_WIDTH}] ${MODELPARAM_VALUE.BANK0_ROUNDTRIP_WIDTH}
}

proc update_MODELPARAM_VALUE.DMA_INIT_TASK_CNT { MODELPARAM_VALUE.DMA_INIT_TASK_CNT PARAM_VALUE.DMA_INIT_TASK_CNT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DMA_INIT_TASK_CNT}] ${MODELPARAM_VALUE.DMA_INIT_TASK_CNT}
}

proc update_MODELPARAM_VALUE.DMA_EXEC_TASK_CNT { MODELPARAM_VALUE.DMA_EXEC_TASK_CNT PARAM_VALUE.DMA_EXEC_TASK_CNT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DMA_EXEC_TASK_CNT}] ${MODELPARAM_VALUE.DMA_EXEC_TASK_CNT}
}


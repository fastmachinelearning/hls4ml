if {[go -compare last new]<0} { solution new -state initial }
solution file add "[file rootname [info script]].yml" -type YAML
if {![gui exists]} { go new }

create_project mgs_ip_prj . -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
add_files -norecurse ../magic_streamer_grp_src/streamGrp.v
update_compile_order -fileset sources_1
add_files -norecurse ../magic_streamer/src/magicStreamer.v




ipx::package_project -root_dir ../magic_streamer_grp_ip -vendor user.org -library user -taxonomy /UserIP -import_files


set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]

ipx::save_core [ipx::current_core]
set_property  ip_repo_paths  ../magic_streamer_grp_ip [current_project]
update_ip_catalog

close_project
exit

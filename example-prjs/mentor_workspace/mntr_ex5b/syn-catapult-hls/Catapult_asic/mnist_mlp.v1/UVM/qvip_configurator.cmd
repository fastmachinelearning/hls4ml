# Create DUT using RTL name
"Configurator" create module DUT top/DUT

"Configurator" change test axi4stream_qvip

# Configure clock/reset
# change clock <clkname> <timescale> <phaseshift> <initialvalue> <hightime> <lowtime>
change clock default_clk_gen ns 0 0 1 1
# change reset <rstname> <startedge> <activecycles> <?> <?> <activeH/L> <syncedge> <sync2clk>
change reset default_reset_gen ns 1 1 1 1 0 1 1

"Configurator" generate
exit yes

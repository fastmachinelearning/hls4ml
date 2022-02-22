# Expose UART Interface on Pmod Header JA
# You may need https://www.sparkfun.com/products/9873

# RX uart, PMOD A pin 2 (JA2), IO_L4P_T0_15, B11, BROWN cable
set_property -dict { PACKAGE_PIN B11 IOSTANDARD LVCMOS33 } [get_ports { pmod_uart_rxd }];

# TX uart, PMOD A pin 3 (JA3), IO_L4N_T0_15, A11, RED cable
set_property -dict { PACKAGE_PIN A11 IOSTANDARD LVCMOS33 } [get_ports { pmod_uart_txd }];

library ieee;
use ieee.std_logic_1164.all ;

entity vhdl_dut is
   port ( clk : in std_logic ;
          rst : in std_logic ;
          in_signal : in std_logic ;
          out_signal :out std_logic 
        );
end vhdl_dut;

architecture rtl of vhdl_dut is
   begin
      P1: process
            variable out_signal_o : std_logic;
            begin
               wait until clk'event and clk = '1';
               out_signal_o := in_signal;
               out_signal <= out_signal_o;
     end process;
   end rtl;

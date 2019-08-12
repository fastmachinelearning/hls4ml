library ieee;
use ieee.std_logic_1164.all;

library BDT;
use BDT.Constants.all;
use BDT.Types.all;

entity testbench is
end testbench;

architecture rtl of testbench is
  signal X : txArray(nFeatures - 1 downto 0) := (others => to_tx(0));
  signal y : tyArray(nClasses - 1 downto 0) := (others => to_ty(0));
  signal clk : std_logic := '0';
begin
    clk <= not clk after 2.5 ns;

    Input : entity work.SimulationInput
    port map(clk, X);

    UUT : entity BDT.BDTTop
    port map(clk, X, y);

    Output : entity work.SimulationOutput
    port map(clk, y);

end architecture rtl;

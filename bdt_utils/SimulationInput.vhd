library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
use std.textio.all;

library BDT;
use BDT.Types.all;
use BDT.Constants.all;

entity SimulationInput is
  generic(
    FileName : string := "./SimulationInput.txt";
    FilePath : string := "./"
  );
  port(
    clk    : in std_logic;
    X : out txArray(nFeatures - 1 downto 0) := (others => to_tx(0))
  );
end SimulationInput;
-- -------------------------------------------------------------------------
-- -------------------------------------------------------------------------
architecture rtl of SimulationInput is

  type tIntArray is array(integer range <>) of integer;

begin
-- pragma synthesis_off
  process(clk)
    file f     : text open read_mode is FilePath & FileName;
    variable s : line;
    variable XRead : tIntArray(X'left downto X'right) := (others => 0);
    variable space : character;
  begin
  if rising_edge(clk) then
    if(not endfile(f)) then
      readline(f, s); 
      for i in  X'range loop
        read(s, XRead(i));
        X(i) <= to_tx(XRead(i));
        if i /= X'right then
          read(s, space);
        end if;
      end loop;
    else
      for i in X'range loop
        X(i) <= to_tx(0);
      end loop;
    end if;
  end if;
  end process;
-- pragma synthesis_on    
end architecture rtl;

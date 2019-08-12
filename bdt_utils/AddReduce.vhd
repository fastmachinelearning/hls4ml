library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

entity AddReduce is
generic(
  id : string := "0"
);
port(
  clk : in std_logic := '0';
  d : in tyArray;
  q : out tyArray
);
end AddReduce;

architecture behavioral of AddReduce is

constant len : integer := d'length;
constant intLen : integer := 2 * ((len + 1) / 2);
constant qLen : integer := (len + 1) / 2;

component AddReduce is
generic(
  id : string := "0"
);
port(
  clk : in std_logic := '0';
  d : in tyArray; --(0 to intLen / 2 - 1);
  q : out tyArray --(0 to qLen / 2 - 1) 
);
end component AddReduce;

begin

G1 : if d'length <= 1 generate
    q <= d;
end generate;

G2 : if d'length = 2 generate
    process(clk)
    begin
        if rising_edge(clk) then
            q(q'left) <= d(d'left) + d(d'right);
        end if;
    end process;
end generate;

GN : if d'length > 2 generate
  -- Lengths are rounded up to nearest even
  signal dInt : tyArray(0 to intLen - 1) := (others => (others => '0'));
  signal qInt : tyArray(0 to qLen - 1) := (others => (others => '0'));
  begin
    dInt(0 to d'length - 1) <= d;

    GNComps:
    for i in 0 to qLen - 1 generate
        Comp:
        process(clk)
        begin
        if rising_edge(clk) then
            q(q'left) <= d(d'left) + d(d'right);
        end if;
        end process;
    end generate;

    Reduce2 : AddReduce
    generic map(id => id & "_C")
    port map(clk, qInt, q);

end generate;

end behavioral;

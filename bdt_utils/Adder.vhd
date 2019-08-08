library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library BDT;
use BDT.Constants.all;
use BDT.Types.all;

entity Adder is
    port(
      clk : in std_logic := '0';
      yin : in  tyArray := (others => to_ty(0));
      yout : out ty
    );
end Adder;

architecture rtl of Adder is
    signal size : natural := yin'left - yin'right;
begin

add_tree1 : if size = 1 generate
    yout <= yin(yin'left);
end generate add_tree1;

add_tree2 : if size = 2 generate
    addproc : process(clk)
    begin
        yout <= yin(yin'left) + yin(yin'right);
    end process;
end generate add_tree2;

add_tree3 : if size > 2 generate
    signal nMid : natural := (yin'length + 1) / 2 + yin'right;
    signal yl : ty := to_ty(0);
    signal yr : ty := to_ty(0);
    begin
    ltree : entity Adder(clk => clk, yin => yin(yin'left downto nMid), yout => yl);
    rtree : entity Adder(clk => clk, yin => yin(nMid-1 downto yin'right), yout => yr);
    yout <= yl + yr;
end generate add_tree3;

end rtl;

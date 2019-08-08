library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library BDT;
use BDT.Constants.all;
use BDT.Types.all;
use BDT.Tree;

entity BDT is
  generic(
    iFeature : intArray2D(0 to nTrees-1)(0 to nNodes-1);
    iChildLeft : intArray2D(0 to nTrees-1)(0 to nNodes-1);
    iChildRight : intArray2D(0 to nTrees-1)(0 to nNodes-1);
    iParent : intArray2D(0 to nTrees-1)(0 to nNodes-1);
    iLeaf : intArray2D(0 to nTrees-1)(0 to nLeaves-1);
    threshold : txArray2D(0 to nTrees-1)(0 to nNodes-1);
    value : tyArray2D(0 to nTrees-1)(0 to nNodes-1);
    reuse : integer := 1
  );
  port(
    clk : in std_logic;  -- clock
    X : in txArray(nFeatures-1 downto 0);           -- input features
    y : out ty           -- output score
  );
end BDT;

architecture rtl of BDT is
  signal yTrees : tyArray(nTrees-1 downto 0); -- The score output by each tree
begin

  -- Make all the tree instances
  TreeGen: for i in 0 to nTrees-1 generate
    Treei : entity Tree
    generic map(
      iFeature => iFeature(i),
      iChildLeft => iChildLeft(i),
      iChildRight => iChildRight(i),
      iParent => iParent(i),
      iLeaf => iLeaf(i),
      threshold => threshold(i),
      value => value(i),
      reuse => reuse
    )port map(clk => clk, X => X, y => yTrees(i));
  end generate;

  -- Sum the output scores using the add tree-reduce
  /* process(clk)
  begin
    if rising_edge(clk) then
      y <= addReduce(yTrees);
    end if;
  end process; */
  
  AddTree : entity BDT.Adder(clk => clk, yin => yTrees, yout => y);

end rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;

entity Tree is
  generic(
    iFeature : intArray(0 to nNodes-1);
    iChildLeft : intArray(0 to nNodes-1);
    iChildRight : intArray(0 to nNodes-1);
    iParent : intArray(0 to nNodes-1);
    iLeaf : intArray(0 to nLeaves-1);
    depth : intArray(0 to nNodes-1);
    threshold : txArray(0 to nNodes-1);
    value : tyArray(0 to nNodes-1);
    reuse : integer := 1
  );
  port(
    clk : in std_logic;  -- clock
    X : in txArray(nFeatures-1 downto 0) := (others => to_tx(0));           -- input features
    y : out ty := to_ty(0)           -- output score
  );
end tree;

architecture rtl of tree is

  signal comparison : boolArray(0 to nNodes-1) := (others => false);
  signal comparisonPipe : boolArray2D(0 to maxdepth)(0 to nNodes-1) := (others => (others => false));
  signal activation : boolArray(0 to nNodes-1) := (others => false);
  signal counter : natural range 0 to reuse-1 := 0;

begin

  -- do all the comparisons
  GenComp:
  for i in 0 to nNodes-1 generate
    NonLeaf: if iFeature(i) /= -2 generate
      process(clk)
      begin
        -- Compare feature for this node to threshold for this node
        if rising_edge(clk) then
          comparison(i) <= X(iFeature(i)) <= threshold(i);
        end if;
      end process;
    end generate NonLeaf;
    -- Leaf nodes don't do comparisons
    Leaf: if iFeature(i) = -2 generate
      process(clk)
      begin
        -- Leaves are always active
        comparison(i) <= true;
      end process;
    end generate Leaf;
  end generate GenComp;

  -- Pipeline the comparisons
  comparisonPipe(0) <= comparison;
  process(clk)
  begin
    if rising_edge(clk) then
      comparisonPipe(1 to maxdepth) <= comparisonPipe(0 to maxdepth-1);
    end if;
  end process;

  -- do all the node activations
  -- the root node is always active
  activation(0) <= true; 
  GenAct:
  for i in 1 to nNodes-1 generate
    -- the root node is always active
    LeftChild:
    if i = iChildLeft(iParent(i)) generate
      process(clk)
      begin
        if rising_edge(clk) then
          activation(i) <= comparisonPipe(depth(i))(iParent(i)) and activation(iParent(i));
        end if;
      end process;    
    end generate LeftChild;
    RightChild:
    if i = iChildRight(iParent(i)) generate
      process(clk)
      begin
        if rising_edge(clk) then
          activation(i) <= (not comparisonPipe(depth(i))(iParent(i))) and activation(iParent(i));
        end if;
      end process;    
    end generate RightChild;
  end generate GenAct;

  -- Assign the score from the active leaf
  GenScore:
  process(clk)
  begin
    if rising_edge(clk) then
     for i in 0 to nLeaves-1 loop
        if activation(iLeaf(i)) then
          y <= value(iLeaf(i));
          exit;
        end if;
      end loop;
    end if;
  end process;

end rtl;

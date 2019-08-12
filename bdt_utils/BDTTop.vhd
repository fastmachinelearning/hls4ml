library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;
use work.Types.all;
-- include arrays

entity BDTTop is
  port(
    clk : in std_logic;  -- clock
    X : in txArray(nFeatures-1 downto 0);           -- input features
    y : out tyArray(nClasses-1 downto 0)            -- output score
  );
end BDTTop;

architecture rtl of BDTTop is
begin

-- instantiate BDTs

end rtl;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

package Constants is

-- hls4ml_insert_constants

  function to_tx(x : integer) return tx;
  function to_ty(y : integer) return ty;

end package;

package body Constants is

  function to_tx(x : integer) return tx is
  begin
    return to_signed(x, tx'length);
  end to_tx;

  function to_ty(y : integer) return ty is
  begin
    return to_signed(y, ty'length);
  end to_ty;

end package body;

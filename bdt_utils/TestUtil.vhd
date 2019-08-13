library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
use std.textio.all;

library BDT;
use BDT.Constants.all;

library TestUtil;

package DataType is 

  type tData is record
    data : ty;
    DataValid : boolean;
  end record;

  constant cNull : tData := ( (others => '0'), false);

  function WriteHeader return string;
  function WriteData(x : tData) return string;

end DataType;

package body DataType is

  function WriteHeader return string is
    variable aLine : line;
  begin
    write(aLine, string'("Data"), right, 15);
    return aLine.all;
  end WriteHeader;

  function WriteData(x : tData) return string is
    variable aLine : line;
  begin
    write(aLine, to_integer(x.data), right, 15);
    return aLine.all;
  end WriteData;

end DataType;

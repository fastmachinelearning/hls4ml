library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

library work;
use work.Constants.all;

package Types is

  type intArray is array(natural range <>) of integer;
  type boolArray is array(natural range <>) of boolean;
  type txArray is array(natural range <>) of tx; -- tx defined in Constants
  type tyArray is array(natural range <>) of ty; -- ty defined in Constants

  type intArray2D is array(natural range <>) of intArray;
  type boolArray2D is array(natural range <>) of boolArray;
  type txArray2D is array(natural range <>) of txArray;
  type tyArray2D is array(natural range <>) of tyArray;

  function addReduce(y : in tyArray) return ty;
  function to_tyArray(yArray : intArray) return tyArray;
  function to_tyArray2D(yArray2D : intArray2D) return tyArray2D;
  function to_txArray(xArray : intArray) return txArray;
  function to_txArray2D(xArray2D : intArray2D) return txArray2D;
  
end Types;

package body Types is
  
  function addReduce(y : in tyArray) return ty is
    -- Sum an array using tree reduce
    -- Recursively build trees of decreasing size
    -- When the size is 2, sum them
    variable ySum : ty := to_ty(0);
    variable lTree, rTree : ty := to_ty(0);
    variable nMid : natural;
  begin
    if y'length = 1 then
      ySum := y(y'left);
    elsif y'length = 2 then
      ySum := y(y'left) + y(y'right);
    else
      -- Find the halfway point
      nMid := (y'length + 1) / 2 + y'right;
      -- Sum each half separately with this function
      rTree := addReduce(y(y'left downto nMid));
      lTree := addReduce(y(nMid-1 downto y'right));
      ySum := ltree + rtree;
    end if;
    return ySum;
  end addReduce;
  
  function to_tyArray(yArray : intArray) return tyArray is
    variable yArrayCast : tyArray(yArray'left downto yArray'right);
  begin
    for i in yArray'right to yArray'left loop
        yArrayCast(i) := to_ty(yArray(i));
    end loop;
    return yArrayCast;
  end to_tyArray;
  
    function to_tyArray2D(yArray2D : intArray2D) return tyArray2D is
    variable yArray2DCast : tyArray2D(yArray2D'left downto yArray2D'right)(yArray2D(0)'left downto yArray2D(0)'right);
  begin
    for i in yArray2D'right to yArray2D'left loop
        yArray2DCast(i) := to_tyArray(yArray2D(i));
    end loop;
    return yArray2DCast;
  end to_tyArray2D;
  
    function to_txArray(xArray : intArray) return txArray is
    variable xArrayCast : txArray(xArray'left downto xArray'right);
  begin
    for i in xArray'right to xArray'left loop
        xArrayCast(i) := to_tx(xArray(i));
    end loop;
    return xArrayCast;
  end to_txArray;
  
    function to_txArray2D(xArray2D : intArray2D) return txArray2D is
    variable xArray2DCast : txArray2D(xArray2D'left downto xArray2D'right)(xArray2D(0)'left downto xArray2D(0)'right);
  begin
    for i in xArray2D'right to xArray2D'left loop
        xArray2DCast(i) := to_txArray(xArray2D(i));
    end loop;
    return xArray2DCast;
  end to_txArray2D;
  
end Types;

-- #########################################################################
-- #########################################################################
-- ###                                                                   ###
-- ###   Use of this code, whether in its current form or modified,      ###
-- ###   implies that you consent to the terms and conditions, namely:   ###
-- ###    - You acknowledge my contribution                              ###
-- ###    - This copyright notification remains intact                   ###
-- ###                                                                   ###
-- ###   Many thanks,                                                    ###
-- ###     Dr. Andrew W. Rose, Imperial College London, 2018             ###
-- ###                                                                   ###
-- #########################################################################
-- #########################################################################

-- .library Utilities

-- -------------------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.STD_LOGIC_TEXTIO.ALL;
-- -------------------------------------------------------------------------

-- -------------------------------------------------------------------------
PACKAGE debugging IS
  CONSTANT Path                 : STRING            := "../../../../DebugFiles/";
  SIGNAL SimulationClockCounter : INTEGER           := -1;
  SIGNAL TimeStamp              : STRING( 1 TO 28 ) := ( OTHERS => '-' );
END PACKAGE debugging;

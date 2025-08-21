module MagicSeqTop #(

    parameter GLOB_ADDR_WIDTH = 32, // Address width for AXI interface
    parameter GLOB_DATA_WIDTH = 32, // Data width for AXI interface

    parameter ADDR_WIDTH = 16, // Address width for AXI interface
    parameter DATA_WIDTH = 32, // Data width for AXI interface

    parameter BANK1_INDEX_WIDTH    =  3, // 2 ^ 2 = 4 slots
    parameter BANK1_SRC_ADDR_WIDTH = 32,
    parameter BANK1_SRC_SIZE_WIDTH = 26,
    parameter BANK1_DST_ADDR_WIDTH = 32,
    parameter BANK1_DST_SIZE_WIDTH = 26,
    parameter BANK1_STATUS_WIDTH   =  2,
    parameter BANK1_PROFILE_WIDTH  = 32,
    parameter BANK1_LD_MSK_WIDTH   =  8,
    parameter BANK1_ST_MSK_WIDTH   =  8,

    parameter BANK0_CONTROL_WIDTH    = 4,
    parameter BANK0_STATUS_WIDTH     = 4,
    parameter BANK0_CNT_WIDTH        = BANK1_INDEX_WIDTH, /// the counter for the sequencer    the BANK1_INDEX_WIDTH SHOULD BE EQUAL TO BANK0_CNT_WIDTH
    parameter BANK0_INTR_WIDTH       = 1, /// the round counter for the sequencer
    parameter BANK0_ROUNDTRIP_WIDTH  = 16, /// the round trip counter for the sequencer


    parameter DMA_INIT_TASK_CNT   = 8, //// (baseAddr0 + size0) + (baseAddr1 + size1)
    parameter DMA_EXEC_TASK_CNT   = 1

) (

input wire clk,
input  wire reset,
/**
PS ------------ AXI4-Lite slave Read Interface 
it is supposed to connect to the PS
*/

// Read Address Channel
input  wire [ADDR_WIDTH-1:0]  S_AXI_ARADDR,
input  wire                   S_AXI_ARVALID,
output wire                   S_AXI_ARREADY,

// Read Data Channel
output wire  [DATA_WIDTH-1:0]   S_AXI_RDATA, ////// read data output acctually it is a reg
output wire [1:0]              S_AXI_RRESP,
output wire                    S_AXI_RVALID,
input  wire                    S_AXI_RREADY,

/**
PS ------------ AXI4-Lite slave write Interface 
it is supposed to connect to the PS
*/

// AXI Lite Write Address Channel
input  wire [ADDR_WIDTH-1:0]  S_AXI_AWADDR,
input  wire                   S_AXI_AWVALID,
output wire                   S_AXI_AWREADY,

// AXI Lite Write Data Channel
input  wire [DATA_WIDTH-1:0]  S_AXI_WDATA,
input  wire [(DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
input  wire                   S_AXI_WVALID,
output wire                   S_AXI_WREADY,

// AXI Lite Write Response Channel
output wire [1:0]             S_AXI_BRESP,
output wire                   S_AXI_BVALID,
input  wire                   S_AXI_BREADY,

/**
 ------------ AXI4-Lite master read Interface 
it is supposed to connect to the PS
*/
// Read Address Channel
output  wire [GLOB_ADDR_WIDTH-1:0]  M_AXI_ARADDR,
output  wire                        M_AXI_ARVALID,
input   wire                        M_AXI_ARREADY,

// Read Data Channel
input   wire  [GLOB_ADDR_WIDTH-1:0]   M_AXI_RDATA, ////// read data output acctually it is a reg
input   wire  [1:0]                   M_AXI_RRESP,
input   wire                          M_AXI_RVALID,
output  wire                          M_AXI_RREADY,

/**
 ------------ AXI4-Lite master write Interface 
it is supposed to connect to the PS
*/

output  wire [GLOB_ADDR_WIDTH-1:0]    M_AXI_AWADDR, ///// actually it is wire
output  wire                          M_AXI_AWVALID,
input   wire                          M_AXI_AWREADY,

    // AXI Lite Write Data Channel
output  wire[GLOB_DATA_WIDTH-1:0]     M_AXI_WDATA, ///// actually it is wire
output  wire[(GLOB_DATA_WIDTH/8)-1:0] M_AXI_WSTRB,
output  wire                          M_AXI_WVALID,
input   wire                          M_AXI_WREADY,

    // AXI Lite Write Response Channel
input  wire [1:0]                     M_AXI_BRESP,
input  wire                           M_AXI_BVALID,
output wire                           M_AXI_BREADY,

output wire [(1 <<BANK0_CNT_WIDTH)-1: 0] slaveReprog,  ///// trigger slave dma to reprogram
input  wire                           nslaveReset,  ///// the slave dma is ready to reprogram

    // communicate with magic streammer
output wire [BANK1_ST_MSK_WIDTH-1: 0] slaveMgsStoreReset, //// reset the interrupt signal
output wire [BANK1_LD_MSK_WIDTH-1: 0] slaveMgsLoadReset,  //// reset the interrupt signal
output wire [BANK1_ST_MSK_WIDTH-1: 0] slaveMgsStoreInit,  //// init the store of the magic streamer bucket
output wire [BANK1_LD_MSK_WIDTH-1: 0] slaveMgsLoadInit,   //// init the load of the magic streamer bucket

input wire  [BANK1_ST_MSK_WIDTH   -1: 0] mgsFinExec, ///// the slave magic sequencer Ip acknowledge that it is finish


output wire [DMA_INIT_TASK_CNT-1: 0]  dbg_slaveInit, ///// trigger slave dma to do somthing
output wire [DMA_INIT_TASK_CNT-1: 0]  dbg_slaveFinInit,

output wire [DMA_EXEC_TASK_CNT-1: 0] dbg_slaveStartExec,
output wire [DMA_EXEC_TASK_CNT-1: 0] dbg_slaveStartExecAccept, ///// the slave dma is ready to start

// controller signal
input wire hw_ctrl_start,
input wire hw_intr_clear,
output wire hw_intr

);


    ///////////////////////////////////////////////////////////
    // internal wiring ////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////
    // outsider interface bank0 (typically from PS)///////////
    //////////////////////////////////////////////////////////

    // setter from outsider
    wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_inp_index;
    wire [BANK1_SRC_ADDR_WIDTH -1:0] ext_bank1_inp_src_addr;
    wire [BANK1_SRC_SIZE_WIDTH -1:0] ext_bank1_inp_src_size;
    wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_inp_des_addr;
    wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_inp_des_size;
    wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_inp_status;
    wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_inp_profile;
    wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_inp_ld_mask;
    wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_mask;
    wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_intr_mask_abs;

    wire ext_bank1_set_src_addr;
    wire ext_bank1_set_src_size;
    wire ext_bank1_set_des_addr;
    wire ext_bank1_set_des_size;
    wire ext_bank1_set_status;
    wire ext_bank1_set_profile;
    wire ext_bank1_set_ld_mask;
    wire ext_bank1_set_st_mask;
    wire ext_bank1_set_st_intr_mask_abs;

    wire ext_bank1_set_fin_src_addr;   /// the result external setting 
    wire ext_bank1_set_fin_src_size;   /// the result external setting 
    wire ext_bank1_set_fin_des_addr;   /// the result external setting 
    wire ext_bank1_set_fin_des_size;   /// the result external setting 
    wire ext_bank1_set_fin_status;     /// the result external setting   
    wire ext_bank1_set_fin_profile;    /// the result external setting  
    wire ext_bank1_set_fin_ld_mask;
    wire ext_bank1_set_fin_st_mask;
    wire ext_bank1_set_fin_st_intr_mask_abs;

    // retriever from outsider
    wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_out_index;
    wire                             ext_bank1_out_req;
    wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_src_addr;
    wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_src_size;
    wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_des_addr;
    wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_des_size;
    wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_out_status;
    wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_out_profile;
    wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_out_ld_mask;     // actually it is a reg
    wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_mask;
    wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_intr_mask;

    wire                             ext_bank1_out_ready;


    //////////////////////////////////////////////////////////
    // outsider interface bank0 (typically from PS)///////////
    //////////////////////////////////////////////////////////
    wire [BANK0_CONTROL_WIDTH-1:0] ext_bank0_inp_control; /// set control data
    wire                           ext_bank0_set_control; /// set control signal

    wire [BANK0_STATUS_WIDTH-1:0] ext_bank0_out_status;  /// read only and it is reg

    wire [BANK0_CNT_WIDTH   -1:0] ext_bank0_out_mainCnt;     /// read only

    wire [BANK0_CNT_WIDTH-1:0]    ext_bank0_inp_endCnt;     ///
    wire                          ext_bank0_set_endCnt;     ///
    wire [BANK0_CNT_WIDTH-1:0]    ext_bank0_out_endCnt;     /// read only

    wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_inp_dmaBaseAddr;
    wire                          ext_bank0_set_dmaBaseAddr;
    wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_out_dmaBaseAddr;

    wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_inp_dfxCtrlAddr;
    wire                          ext_bank0_set_dfxCtrlAddr;
    wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_out_dfxCtrlAddr;

    wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intrEna; //// input data for the interrupt counter
    wire                          ext_bank0_set_intrEna; //// set the interrupt counter ONLY when the system is in shutdown state
    wire[BANK0_INTR_WIDTH-1: 0]   ext_bank0_out_intrEna; //// output data for the interrupt counter

    wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intr; //// input data for the interrupt counter
    wire                          ext_bank0_set_intr; //// set the interrupt counter ONLY when the system is in shutdown state
    wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_out_intr; //// output data for the interrupt counter
    ///// assign for the output signal
    assign hw_intr = ext_bank0_out_intr[0]; /// the first bit is the interrupt signal

    wire [BANK0_ROUNDTRIP_WIDTH-1: 0]  ext_bank0_inp_roundTrip; /// input data for the round trip counter
    wire                               ext_bank0_set_roundTrip; /// set the round trip counter ONLY when the system is in shutdown state
    wire [BANK0_ROUNDTRIP_WIDTH-1: 0]  ext_bank0_out_roundTrip; /// output data for the round trip counter    
    //////////////////////////////////////////////////////////
    // slave functionality                         ///////////
    //////////////////////////////////////////////////////////

    ////// the mgs control will be handle by io
    wire [DMA_INIT_TASK_CNT-1: 0] slaveInit   ; ///// trigger slave dma to do somthing
    wire [DMA_INIT_TASK_CNT-1: 0] slaveFinInit;

    wire [DMA_EXEC_TASK_CNT-1: 0] slaveStartExec      ;
    wire [DMA_EXEC_TASK_CNT-1: 0] slaveStartExecAccept; ///// the slave dma is ready to start

    assign dbg_slaveInit            = slaveInit;
    assign dbg_slaveFinInit         = slaveFinInit;
    assign dbg_slaveStartExec       = slaveStartExec;
    assign dbg_slaveStartExecAccept = slaveStartExecAccept;

    /// wire  slaveFinExec; ///// the slave dma is finished, so we can go to triggering next
    //////////// we route it as input

    wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_src_addr;      // actually it is a reg
    wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_src_size;      // actually it is a reg
    wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_des_addr;
    wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_des_size;
    wire [BANK1_STATUS_WIDTH   -1:0] slave_bank1_out_status  ;      // actually it is a reg
    wire [BANK1_PROFILE_WIDTH  -1:0] slave_bank1_out_profile ;      // actually it is a reg
    wire [BANK1_LD_MSK_WIDTH   -1:0] slave_bank1_out_ld_mask;     // actually it is a reg
    wire [BANK1_ST_MSK_WIDTH   -1:0] slave_bank1_out_st_mask;
    wire [BANK1_ST_MSK_WIDTH   -1:0] slave_bank1_out_st_intr_mask;


    ////// for now we dummy assign the dma connection signal to prevent errror

///////////////////////////////////////////////////////////////
///////// create axi interface for read channel ///////////////
///////////////////////////////////////////////////////////////

s_axi_read #(
    .GLOB_ADDR_WIDTH(GLOB_ADDR_WIDTH),
    .GLOB_DATA_WIDTH(GLOB_DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),

    .BANK1_INDEX_WIDTH(BANK1_INDEX_WIDTH),
    .BANK1_SRC_ADDR_WIDTH(BANK1_SRC_ADDR_WIDTH),
    .BANK1_SRC_SIZE_WIDTH(BANK1_SRC_SIZE_WIDTH),
    .BANK1_DST_ADDR_WIDTH(BANK1_DST_ADDR_WIDTH),
    .BANK1_DST_SIZE_WIDTH(BANK1_DST_SIZE_WIDTH),
    .BANK1_STATUS_WIDTH(BANK1_STATUS_WIDTH),
    .BANK1_PROFILE_WIDTH(BANK1_PROFILE_WIDTH),
    .BANK1_LD_MSK_WIDTH(BANK1_LD_MSK_WIDTH),
    .BANK1_ST_MSK_WIDTH(BANK1_ST_MSK_WIDTH),

    .BANK0_CONTROL_WIDTH(BANK0_CONTROL_WIDTH),
    .BANK0_STATUS_WIDTH(BANK0_STATUS_WIDTH),
    .BANK0_CNT_WIDTH(BANK0_CNT_WIDTH),
    .BANK0_INTR_WIDTH(BANK0_INTR_WIDTH),
    .BANK0_ROUNDTRIP_WIDTH(BANK0_ROUNDTRIP_WIDTH)
) ps_axi_reader(
    .clk(clk),
    .reset(reset),

    // Read Address Channel
    .S_AXI_ARADDR(S_AXI_ARADDR),
    .S_AXI_ARVALID(S_AXI_ARVALID),
    .S_AXI_ARREADY(S_AXI_ARREADY),

    // Read Data Channel
    .S_AXI_RDATA(S_AXI_RDATA), ////// read data output acctually it is a reg
    .S_AXI_RRESP(S_AXI_RRESP),
    .S_AXI_RVALID(S_AXI_RVALID),
    .S_AXI_RREADY(S_AXI_RREADY),

    ////// bank1 interconnect
    .ext_bank1_out_index(ext_bank1_out_index),
    .ext_bank1_out_req(ext_bank1_out_req),           // actually it is a wire
    .ext_bank1_out_src_addr(ext_bank1_out_src_addr),      // actually it is a wire
    .ext_bank1_out_src_size(ext_bank1_out_src_size),      // actually it is a wire
    .ext_bank1_out_des_addr(ext_bank1_out_des_addr),
    .ext_bank1_out_des_size(ext_bank1_out_des_size),
    .ext_bank1_out_status(ext_bank1_out_status  ),      // actually it is a wire
    .ext_bank1_out_profile(ext_bank1_out_profile),       // actually it is a wire
    .ext_bank1_out_ld_mask(ext_bank1_out_ld_mask),
    .ext_bank1_out_st_mask(ext_bank1_out_st_mask),
    .ext_bank1_out_st_intr_mask(ext_bank1_out_st_intr_mask),
    .ext_bank1_out_ready(ext_bank1_out_ready),         // actually it is a wire


    ////// bank0 interconnect
    .ext_bank0_out_status(ext_bank0_out_status),  /// read only and it is reg
    .ext_bank0_out_mainCnt(ext_bank0_out_mainCnt),     /// read only
    .ext_bank0_out_endCnt(ext_bank0_out_endCnt),     /// read only
    .ext_bank0_out_dmaBaseAddr(ext_bank0_out_dmaBaseAddr),
    .ext_bank0_out_dfxCtrlAddr(ext_bank0_out_dfxCtrlAddr),
    .ext_bank0_out_intrEna(ext_bank0_out_intrEna),
    .ext_bank0_out_intr(ext_bank0_out_intr),
    .ext_bank0_out_roundTrip(ext_bank0_out_roundTrip) /// output data for the round trip counter

);

//////////////////////////////////////////////////////////////////////
///////// create axi slave interface for write channel ///////////////
//////////////////////////////////////////////////////////////////////

s_axi_write #(
    .GLOB_ADDR_WIDTH(GLOB_ADDR_WIDTH),
    .GLOB_DATA_WIDTH(GLOB_DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),

    .BANK1_INDEX_WIDTH(BANK1_INDEX_WIDTH),
    .BANK1_SRC_ADDR_WIDTH(BANK1_SRC_ADDR_WIDTH),
    .BANK1_SRC_SIZE_WIDTH(BANK1_SRC_SIZE_WIDTH),
    .BANK1_DST_ADDR_WIDTH(BANK1_DST_ADDR_WIDTH),
    .BANK1_DST_SIZE_WIDTH(BANK1_DST_SIZE_WIDTH),
    .BANK1_STATUS_WIDTH(BANK1_STATUS_WIDTH),
    .BANK1_PROFILE_WIDTH(BANK1_PROFILE_WIDTH),
    .BANK1_LD_MSK_WIDTH(BANK1_LD_MSK_WIDTH),
    .BANK1_ST_MSK_WIDTH(BANK1_ST_MSK_WIDTH),

    .BANK0_CONTROL_WIDTH(BANK0_CONTROL_WIDTH),
    .BANK0_STATUS_WIDTH(BANK0_STATUS_WIDTH),
    .BANK0_CNT_WIDTH(BANK0_CNT_WIDTH),
    .BANK0_INTR_WIDTH(BANK0_INTR_WIDTH),
    .BANK0_ROUNDTRIP_WIDTH(BANK0_ROUNDTRIP_WIDTH)
) ps_axi_writer(
    .clk(clk),
    .reset(reset),

    // AXI Lite Write Address Channel
    .S_AXI_AWADDR(S_AXI_AWADDR),
    .S_AXI_AWVALID(S_AXI_AWVALID),
    .S_AXI_AWREADY(S_AXI_AWREADY),

    // AXI Lite Write Data Channel
    .S_AXI_WDATA(S_AXI_WDATA),
    .S_AXI_WSTRB(S_AXI_WSTRB),
    .S_AXI_WVALID(S_AXI_WVALID),
    .S_AXI_WREADY(S_AXI_WREADY),

    // AXI Lite Write Response Channel
    .S_AXI_BRESP(S_AXI_BRESP),
    .S_AXI_BVALID(S_AXI_BVALID),
    .S_AXI_BREADY(S_AXI_BREADY),

    //// bank1 interconnect
    .ext_bank1_inp_index(ext_bank1_inp_index),       // actually it is a wire
    .ext_bank1_inp_src_addr(ext_bank1_inp_src_addr),  // actually it is a wire
    .ext_bank1_inp_src_size(ext_bank1_inp_src_size),  // actually it is a wire
    .ext_bank1_inp_des_addr(ext_bank1_inp_des_addr),  // actually it is a wire
    .ext_bank1_inp_des_size(ext_bank1_inp_des_size),  // actually it is a wire
    .ext_bank1_inp_status(ext_bank1_inp_status),      // actually it is a wire
    .ext_bank1_inp_profile(ext_bank1_inp_profile),     // actually it is a wire
    .ext_bank1_inp_ld_mask(ext_bank1_inp_ld_mask),
    .ext_bank1_inp_st_mask(ext_bank1_inp_st_mask),
    .ext_bank1_inp_st_intr_mask_abs(ext_bank1_inp_st_intr_mask_abs),

    .ext_bank1_set_src_addr(ext_bank1_set_src_addr),
    .ext_bank1_set_src_size(ext_bank1_set_src_size),
    .ext_bank1_set_des_addr(ext_bank1_set_des_addr),
    .ext_bank1_set_des_size(ext_bank1_set_des_size),
    .ext_bank1_set_status(ext_bank1_set_status),
    .ext_bank1_set_profile(ext_bank1_set_profile),
    .ext_bank1_set_ld_mask(ext_bank1_set_ld_mask),
    .ext_bank1_set_st_mask(ext_bank1_set_st_mask),
    .ext_bank1_set_st_intr_mask_abs(ext_bank1_set_st_intr_mask_abs),
    //// bank0 interconnect
    .ext_bank0_inp_control(ext_bank0_inp_control),
    .ext_bank0_set_control(ext_bank0_set_control),
    .ext_bank0_inp_endCnt(ext_bank0_inp_endCnt),
    .ext_bank0_set_endCnt(ext_bank0_set_endCnt),
    .ext_bank0_inp_dmaBaseAddr(ext_bank0_inp_dmaBaseAddr),
    .ext_bank0_set_dmaBaseAddr(ext_bank0_set_dmaBaseAddr),
    .ext_bank0_inp_dfxCtrlAddr(ext_bank0_inp_dfxCtrlAddr),
    .ext_bank0_set_dfxCtrlAddr(ext_bank0_set_dfxCtrlAddr),
    .ext_bank0_inp_intrEna(ext_bank0_inp_intrEna),
    .ext_bank0_set_intrEna(ext_bank0_set_intrEna),
    .ext_bank0_inp_intr(ext_bank0_inp_intr),
    .ext_bank0_set_intr(ext_bank0_set_intr),
    .ext_bank0_inp_roundTrip(ext_bank0_inp_roundTrip), /// input data for the round trip counter
    .ext_bank0_set_roundTrip(ext_bank0_set_roundTrip)  /// set the round trip counter ONLY when the system is in shutdown state
);


//////////////////////////////////////////////////////////////////////
///////// create axi master interface for read channel //////////////
//////////////////////////////////////////////////////////////////////


m_axi_read #(
    .GLOB_ADDR_WIDTH(GLOB_ADDR_WIDTH),
    .GLOB_DATA_WIDTH(GLOB_DATA_WIDTH),
    .BANK1_INDEX_WIDTH(BANK1_INDEX_WIDTH),
    .BANK1_SRC_ADDR_WIDTH(BANK1_SRC_ADDR_WIDTH),
    .BANK1_SRC_SIZE_WIDTH(BANK1_SRC_SIZE_WIDTH),
    .BANK1_DST_ADDR_WIDTH(BANK1_DST_ADDR_WIDTH),
    .BANK1_DST_SIZE_WIDTH(BANK1_DST_SIZE_WIDTH),
    .BANK1_STATUS_WIDTH(BANK1_STATUS_WIDTH),
    .BANK1_PROFILE_WIDTH(BANK1_PROFILE_WIDTH),
    .BANK1_LD_MSK_WIDTH(BANK1_LD_MSK_WIDTH),
    .BANK1_ST_MSK_WIDTH(BANK1_ST_MSK_WIDTH),

    .BANK0_CONTROL_WIDTH(BANK0_CONTROL_WIDTH),
    .BANK0_STATUS_WIDTH(BANK0_STATUS_WIDTH),
    .BANK0_CNT_WIDTH(BANK0_CNT_WIDTH),
    .DMA_INIT_TASK_CNT(DMA_INIT_TASK_CNT),
    .DMA_EXEC_TASK_CNT(DMA_EXEC_TASK_CNT)

) axiCmdRead(
    .clk(clk),
    .reset(reset),
    .M_AXI_ARADDR(M_AXI_ARADDR),
    .M_AXI_ARVALID(M_AXI_ARVALID),
    .M_AXI_ARREADY(M_AXI_ARREADY),
    .M_AXI_RDATA(M_AXI_RDATA),
    .M_AXI_RRESP(M_AXI_RRESP),
    .M_AXI_RVALID(M_AXI_RVALID),
    .M_AXI_RREADY(M_AXI_RREADY)
);


//////////////////////////////////////////////////////////////////////
///////// create axi master interface for write channel //////////////
//////////////////////////////////////////////////////////////////////

m_axi_write #(
        .GLOB_ADDR_WIDTH(GLOB_ADDR_WIDTH),
    .GLOB_DATA_WIDTH(GLOB_DATA_WIDTH),
    .BANK1_INDEX_WIDTH(BANK1_INDEX_WIDTH),
    .BANK1_SRC_ADDR_WIDTH(BANK1_SRC_ADDR_WIDTH),
    .BANK1_SRC_SIZE_WIDTH(BANK1_SRC_SIZE_WIDTH),
    .BANK1_DST_ADDR_WIDTH(BANK1_DST_ADDR_WIDTH),
    .BANK1_DST_SIZE_WIDTH(BANK1_DST_SIZE_WIDTH),
    .BANK1_STATUS_WIDTH(BANK1_STATUS_WIDTH),
    .BANK1_PROFILE_WIDTH(BANK1_PROFILE_WIDTH),
    .BANK1_LD_MSK_WIDTH(BANK1_LD_MSK_WIDTH),
    .BANK1_ST_MSK_WIDTH(BANK1_ST_MSK_WIDTH),

    .BANK0_CONTROL_WIDTH(BANK0_CONTROL_WIDTH),
    .BANK0_STATUS_WIDTH(BANK0_STATUS_WIDTH),
    .BANK0_CNT_WIDTH(BANK0_CNT_WIDTH),
    .DMA_INIT_TASK_CNT(DMA_INIT_TASK_CNT),
    .DMA_EXEC_TASK_CNT(DMA_EXEC_TASK_CNT)
) axiCmdWrite (

    .clk(clk),
    .reset(reset),

    .M_AXI_AWADDR(M_AXI_AWADDR),
    .M_AXI_AWVALID(M_AXI_AWVALID),
    .M_AXI_AWREADY(M_AXI_AWREADY),

    .M_AXI_WDATA(M_AXI_WDATA),
    .M_AXI_WSTRB(M_AXI_WSTRB),
    .M_AXI_WVALID(M_AXI_WVALID),
    .M_AXI_WREADY(M_AXI_WREADY),

    .M_AXI_BRESP(M_AXI_BRESP),
    .M_AXI_BVALID(M_AXI_BVALID),
    .M_AXI_BREADY(M_AXI_BREADY),

    .ext_bank0_out_dmaBaseAddr(ext_bank0_out_dmaBaseAddr),


    .slaveInit(slaveInit),
    .slaveFinInit(slaveFinInit),

    .slaveStartExec(slaveStartExec),
    .slaveStartExecAccept(slaveStartExecAccept),

    .slave_bank1_out_src_addr(slave_bank1_out_src_addr),
    .slave_bank1_out_src_size(slave_bank1_out_src_size),
    .slave_bank1_out_des_addr(slave_bank1_out_des_addr),
    .slave_bank1_out_des_size(slave_bank1_out_des_size),
    .slave_bank1_out_status(slave_bank1_out_status),
    .slave_bank1_out_profile(slave_bank1_out_profile)

);



//////////////////////////////////////////////////////////////////////
////////////// connect to main core //////////////////////////////////
//////////////////////////////////////////////////////////////////////

MagicSeqCore #(
    .GLOB_ADDR_WIDTH(GLOB_ADDR_WIDTH),
    .GLOB_DATA_WIDTH(GLOB_DATA_WIDTH),
    .BANK1_INDEX_WIDTH(BANK1_INDEX_WIDTH),
    .BANK1_SRC_ADDR_WIDTH(BANK1_SRC_ADDR_WIDTH),
    .BANK1_SRC_SIZE_WIDTH(BANK1_SRC_SIZE_WIDTH),
    .BANK1_DST_ADDR_WIDTH(BANK1_DST_ADDR_WIDTH),
    .BANK1_DST_SIZE_WIDTH(BANK1_DST_SIZE_WIDTH),
    .BANK1_STATUS_WIDTH(BANK1_STATUS_WIDTH),
    .BANK1_PROFILE_WIDTH(BANK1_PROFILE_WIDTH),
    .BANK1_LD_MSK_WIDTH(BANK1_LD_MSK_WIDTH),
    .BANK1_ST_MSK_WIDTH(BANK1_ST_MSK_WIDTH),

    .BANK0_CONTROL_WIDTH(BANK0_CONTROL_WIDTH),
    .BANK0_STATUS_WIDTH(BANK0_STATUS_WIDTH),
    .BANK0_CNT_WIDTH(BANK0_CNT_WIDTH),
    .BANK0_INTR_WIDTH(BANK0_INTR_WIDTH),
    .BANK0_ROUNDTRIP_WIDTH(BANK0_ROUNDTRIP_WIDTH),
    .DMA_INIT_TASK_CNT(DMA_INIT_TASK_CNT),
    .DMA_EXEC_TASK_CNT(DMA_EXEC_TASK_CNT)
) magicSeqCore(
    .clk(clk),
    .reset(reset),

    // setter from outsider
    .ext_bank1_inp_index                (ext_bank1_inp_index),
    .ext_bank1_inp_src_addr             (ext_bank1_inp_src_addr),
    .ext_bank1_inp_src_size             (ext_bank1_inp_src_size),
    .ext_bank1_inp_des_addr             (ext_bank1_inp_des_addr),
    .ext_bank1_inp_des_size             (ext_bank1_inp_des_size),
    .ext_bank1_inp_status               (ext_bank1_inp_status),
    .ext_bank1_inp_profile              (ext_bank1_inp_profile),
    .ext_bank1_inp_ld_mask              (ext_bank1_inp_ld_mask),
    .ext_bank1_inp_st_mask              (ext_bank1_inp_st_mask),
    .ext_bank1_inp_st_intr_mask_abs     (ext_bank1_inp_st_intr_mask_abs),

    .ext_bank1_set_src_addr             (ext_bank1_set_src_addr),
    .ext_bank1_set_src_size             (ext_bank1_set_src_size),
    .ext_bank1_set_des_addr             (ext_bank1_set_des_addr),
    .ext_bank1_set_des_size             (ext_bank1_set_des_size),
    .ext_bank1_set_status               (ext_bank1_set_status),
    .ext_bank1_set_profile              (ext_bank1_set_profile),
    .ext_bank1_set_ld_mask              (ext_bank1_set_ld_mask),
    .ext_bank1_set_st_mask              (ext_bank1_set_st_mask),
    .ext_bank1_set_st_intr_mask_abs     (ext_bank1_set_st_intr_mask_abs),

    .ext_bank1_set_fin_src_addr         (ext_bank1_set_fin_src_addr),
    .ext_bank1_set_fin_src_size         (ext_bank1_set_fin_src_size),
    .ext_bank1_set_fin_des_addr         (ext_bank1_set_fin_des_addr),
    .ext_bank1_set_fin_des_size         (ext_bank1_set_fin_des_size),
    .ext_bank1_set_fin_status           (ext_bank1_set_fin_status),
    .ext_bank1_set_fin_profile          (ext_bank1_set_fin_profile),
    .ext_bank1_set_fin_ld_mask          (ext_bank1_set_fin_ld_mask),
    .ext_bank1_set_fin_st_mask          (ext_bank1_set_fin_st_mask),
    .ext_bank1_set_fin_st_intr_mask_abs (ext_bank1_set_fin_st_intr_mask_abs),
    

    // retrieve to outsider

    .ext_bank1_out_index        (ext_bank1_out_index),
    .ext_bank1_out_req          (ext_bank1_out_req),
    .ext_bank1_out_src_addr     (ext_bank1_out_src_addr),
    .ext_bank1_out_src_size     (ext_bank1_out_src_size),
    .ext_bank1_out_des_addr     (ext_bank1_out_des_addr),
    .ext_bank1_out_des_size     (ext_bank1_out_des_size),
    .ext_bank1_out_status       (ext_bank1_out_status),
    .ext_bank1_out_profile      (ext_bank1_out_profile),
    .ext_bank1_out_ld_mask      (ext_bank1_out_ld_mask),
    .ext_bank1_out_st_mask      (ext_bank1_out_st_mask),
    .ext_bank1_out_st_intr_mask (ext_bank1_out_st_intr_mask),
    .ext_bank1_out_ready        (ext_bank1_out_ready),

    .ext_bank0_inp_control(ext_bank0_inp_control),
    .ext_bank0_set_control(ext_bank0_set_control),
    .hw_ctrl_start(hw_ctrl_start),

    .ext_bank0_out_status(ext_bank0_out_status),
    
    .ext_bank0_out_mainCnt(ext_bank0_out_mainCnt),

    .ext_bank0_inp_endCnt(ext_bank0_inp_endCnt),
    .ext_bank0_set_endCnt(ext_bank0_set_endCnt),
    .ext_bank0_out_endCnt(ext_bank0_out_endCnt),

    .ext_bank0_inp_dmaBaseAddr(ext_bank0_inp_dmaBaseAddr),
    .ext_bank0_set_dmaBaseAddr(ext_bank0_set_dmaBaseAddr),
    .ext_bank0_out_dmaBaseAddr(ext_bank0_out_dmaBaseAddr),

    .ext_bank0_inp_dfxCtrlAddr(ext_bank0_inp_dfxCtrlAddr),
    .ext_bank0_set_dfxCtrlAddr(ext_bank0_set_dfxCtrlAddr),
    .ext_bank0_out_dfxCtrlAddr(ext_bank0_out_dfxCtrlAddr),

    .ext_bank0_inp_intrEna(ext_bank0_inp_intrEna),
    .ext_bank0_set_intrEna(ext_bank0_set_intrEna),
    .ext_bank0_out_intrEna(ext_bank0_out_intrEna),
    
    .ext_bank0_inp_intr(ext_bank0_inp_intr),
    .ext_bank0_set_intr(ext_bank0_set_intr),
    .ext_bank0_out_intr(ext_bank0_out_intr),
    .hw_intr_clear(hw_intr_clear),

    .ext_bank0_inp_roundTrip(ext_bank0_inp_roundTrip),
    .ext_bank0_set_roundTrip(ext_bank0_set_roundTrip),
    .ext_bank0_out_roundTrip(ext_bank0_out_roundTrip),
    /////// communicate with slave

    .slaveReprog(slaveReprog), ///// trigger slave dma to reprogram
    .nslaveReset(nslaveReset), ///// the slave dma is ready to reprogram

    .slaveMgsStoreReset(slaveMgsStoreReset),
    .slaveMgsLoadReset (slaveMgsLoadReset),
    .slaveMgsStoreInit (slaveMgsStoreInit),
    .slaveMgsLoadInit  (slaveMgsLoadInit),
    .slaveInit         (slaveInit), ///// trigger slave dma to do somthing
    .slaveFinInit      (slaveFinInit),

    .mgsFinExec(mgsFinExec), ///// the slave dma is finished, so we can go to triggering next

    .slave_bank1_out_src_addr(slave_bank1_out_src_addr) ,      // actually it is a reg
    .slave_bank1_out_src_size(slave_bank1_out_src_size) ,      // actually it is a reg
    .slave_bank1_out_des_addr(slave_bank1_out_des_addr) ,
    .slave_bank1_out_des_size(slave_bank1_out_des_size) ,
    .slave_bank1_out_status(slave_bank1_out_status)   ,      // actually it is a reg
    .slave_bank1_out_profile(slave_bank1_out_profile)        // actually it is a reg

);

endmodule
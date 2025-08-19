module MagicSeqCore #(

    parameter GLOB_ADDR_WIDTH = 32, // Address width for AXI interface
    parameter GLOB_DATA_WIDTH = 32, // Data width for AXI interface

    parameter BANK1_INDEX_WIDTH    =  3, // 2 ^ 2 = 4 slots
    parameter BANK1_SRC_ADDR_WIDTH = 32,
    parameter BANK1_SRC_SIZE_WIDTH = 26,
    parameter BANK1_DST_ADDR_WIDTH = 32,
    parameter BANK1_DST_SIZE_WIDTH = 26,
    parameter BANK1_STATUS_WIDTH   =  2,
    parameter BANK1_PROFILE_WIDTH  = 32,
    parameter BANK1_LD_MSK_WIDTH    =  8,
    parameter BANK1_ST_MSK_WIDTH    =  8,

    parameter BANK0_CONTROL_WIDTH = 4,
    parameter BANK0_STATUS_WIDTH  = 4,
    parameter BANK0_CNT_WIDTH     = BANK1_INDEX_WIDTH, /// the counter for the sequencer
    parameter BANK0_INTR_WIDTH    = 1, /// the interrupt for the sequencer
    parameter BANK0_ROUNDTRIP_WIDTH = 16, /// the round trip counter for the sequencer

    parameter DMA_INIT_TASK_CNT   = 8, //// (reset interrupt + startReadChannel + baseAddr0 + size0) + (startWriteChannel + baseAddr1 + size1)
    parameter DMA_EXEC_TASK_CNT   = 1
) (
    input wire clk,
    input wire reset,
    //////////////////////////////////////////////////////////
    // outsider interface bank 1 (typically from PS)///////////
    //////////////////////////////////////////////////////////

    ///////// setter from outsider
    input wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_inp_index,
    input wire [BANK1_SRC_ADDR_WIDTH -1:0] ext_bank1_inp_src_addr,
    input wire [BANK1_SRC_SIZE_WIDTH -1:0] ext_bank1_inp_src_size,
    input wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_inp_des_addr,
    input wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_inp_des_size,
    input wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_inp_status,
    input wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_inp_profile,
    input wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_inp_ld_mask,
    input wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_mask,
    // input wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_intr_mask_ack,
    input wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_intr_mask_abs,

    input wire ext_bank1_set_src_addr,
    input wire ext_bank1_set_src_size,
    input wire ext_bank1_set_des_addr,
    input wire ext_bank1_set_des_size,
    input wire ext_bank1_set_status,
    input wire ext_bank1_set_profile,
    input wire ext_bank1_set_ld_mask,
    input wire ext_bank1_set_st_mask,
    // input wire ext_bank1_set_st_intr_mask_ack,
    input wire ext_bank1_set_st_intr_mask_abs,

    output wire ext_bank1_set_fin_src_addr,   /// the result external setting 
    output wire ext_bank1_set_fin_src_size,   /// the result external setting 
    output wire ext_bank1_set_fin_des_addr,   /// the result external setting 
    output wire ext_bank1_set_fin_des_size,   /// the result external setting 
    output wire ext_bank1_set_fin_status,     /// the result external setting   
    output wire ext_bank1_set_fin_profile,    /// the result external setting  
    output wire ext_bank1_set_fin_ld_mask,
    output wire ext_bank1_set_fin_st_mask,
    // output wire ext_bank1_set_fin_st_intr_mask_ack,
    output wire ext_bank1_set_fin_st_intr_mask_abs,

    ///////// send outsider
    
    input  wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_out_index,
    input  wire                             ext_bank1_out_req,           // actually it is a wire
    ///////// 
    output wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_src_addr,      // actually it is a reg
    output wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_src_size,      // actually it is a reg
    output wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_des_addr,
    output wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_des_size,
    output wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_out_status  ,      // actually it is a reg
    output wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_out_profile,       // actually it is a reg
    output wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_out_ld_mask,     // actually it is a reg
    output wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_mask,
    output wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_intr_mask,

    
    
    output wire                             ext_bank1_out_ready,         // actually it is a reg


    //////////////////////////////////////////////////////////
    // outsider interface bank0 (typically from PS)///////////
    //////////////////////////////////////////////////////////
    input wire [BANK0_CONTROL_WIDTH-1:0] ext_bank0_inp_control, /// set control data
    input wire                           ext_bank0_set_control, /// set control signal
    input wire                           hw_ctrl_start,

    output wire [BANK0_STATUS_WIDTH-1:0] ext_bank0_out_status,  /// read only and it is reg

    output wire [BANK0_CNT_WIDTH   -1:0] ext_bank0_out_mainCnt,     /// read only

    input  wire [BANK0_CNT_WIDTH-1:0]    ext_bank0_inp_endCnt,      ///
    input  wire                          ext_bank0_set_endCnt,      ///
    output wire [BANK0_CNT_WIDTH-1:0]    ext_bank0_out_endCnt,      /// read only

    input  wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_inp_dmaBaseAddr,
    input  wire                          ext_bank0_set_dmaBaseAddr,
    output wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_out_dmaBaseAddr,

    input  wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_inp_dfxCtrlAddr,
    input  wire                          ext_bank0_set_dfxCtrlAddr,
    output wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_out_dfxCtrlAddr,

    input  wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intrEna, //// input data for the interrupt counter
    input  wire                          ext_bank0_set_intrEna, //// set the interrupt counter ONLY when the system is in shutdown state
    output wire[BANK0_INTR_WIDTH-1: 0]   ext_bank0_out_intrEna, //// output data for the interrupt counter

    input wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intr, //// input data for the interrupt counter
    input wire                          ext_bank0_set_intr, //// set the interrupt counter ONLY when the system is in shutdown state
    output wire[BANK0_INTR_WIDTH-1: 0]  ext_bank0_out_intr, //// output data for the interrupt counter
    input                               hw_intr_clear, //// clear the interrupt signal

    input wire [BANK0_ROUNDTRIP_WIDTH-1: 0]  ext_bank0_inp_roundTrip, /// input data for the round trip counter
    input wire                               ext_bank0_set_roundTrip, /// set the round trip counter ONLY when the system is in shutdown state
    output wire [BANK0_ROUNDTRIP_WIDTH-1: 0] ext_bank0_out_roundTrip, /// output data for the round trip counter



    //////////////////////////////////////////////////////////
    // slave functionality                         ///////////
    //////////////////////////////////////////////////////////


    ///// 
    ///// this will trigger the dfx Controller usally hardware trigger
    /////
    output wire [(1 <<BANK0_CNT_WIDTH)-1: 0] slaveReprog, ///// trigger slave dfx Controller to reprogram
    input  wire nslaveReset, ///// the signal from dfx Controller that rm module is reset active low
    ///// 
    ///// this will trigger the dma Controller
    /////
    output wire [BANK1_ST_MSK_WIDTH-1: 0] slaveMgsStoreReset, //// reset the interrupt signal
    output wire [BANK1_LD_MSK_WIDTH-1: 0] slaveMgsLoadReset,  //// reset the interrupt signal
    output wire [BANK1_ST_MSK_WIDTH-1: 0] slaveMgsStoreInit,  //// init the store of the magic streamer bucket
    output wire [BANK1_LD_MSK_WIDTH-1: 0] slaveMgsLoadInit,   //// init the load of the magic streamer bucket
    output wire [DMA_INIT_TASK_CNT -1: 0] slaveInit   , ///// trigger slave dma to do somthing via AXI
    input  wire [DMA_INIT_TASK_CNT -1: 0] slaveFinInit, ///// trigger slave dma to do somthing via AXI

    ////// finish exec 
    input wire  [BANK1_ST_MSK_WIDTH   -1: 0] mgsFinExec, ///// the slave magic sequencer Ip acknowledge that it is finish
                                            /////// -1 because we reserve bit 0 for dma

    output wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_src_addr,      // actually it is a reg
    output wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_src_size,      // actually it is a reg
    output wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_des_addr,
    output wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_des_size,
    output wire [BANK1_STATUS_WIDTH   -1:0] slave_bank1_out_status  ,      // actually it is a reg
    output wire [BANK1_PROFILE_WIDTH  -1:0] slave_bank1_out_profile,        // actually it is a reg
    output wire [BANK1_LD_MSK_WIDTH   -1:0] slave_bank1_out_ld_mask,     // actually it is a reg
    output wire [BANK1_ST_MSK_WIDTH   -1:0] slave_bank1_out_st_mask,
    output wire [BANK1_ST_MSK_WIDTH   -1:0] slave_bank1_out_st_intr_mask



);


localparam STATUS_SHUTDOWN       = 4'b0000;
localparam STATUS_REPROG         = 4'b0001;
localparam STATUS_W4SLAVERESET   = 4'b0010;
localparam STATUS_W4SLAVEOP      = 4'b0011;
localparam STATUS_CLEAR_MGS      = 4'b0100;
localparam STATUS_INITIALIZE_MGS = 4'b0101; // initialize magic streamer, to reset magic streamer and start the streaming
localparam STATUS_INITIALIZE_DMA = 4'b0110; // the state will reset the interrupt signal
localparam STATUS_SET_DMA_LOAD   = 4'b0111; // the system is setting the dma load, we can trigger the slave to do something
localparam STATUS_SET_DMA_STORE  = 4'b1000; // the system is setting the dma store, we can trigger the slave to do something
localparam STATUS_TRIGGERING     = 4'b1001;
localparam STATUS_WAIT4FIN       = 4'b1010;
localparam STATUS_PAUSEONERROR   = 4'b1111; // the system is paused on error, we can not do anything


///////////// task for dma
localparam DMA_TASK_RESET_INTR_BEG = 0; // reset the interrupt signal task
localparam DMA_TASK_RESET_INTR_END = 1; // reset the interrupt signal
localparam DMA_TASK_LOAD_TASK_BEG  = 2;  // load task begin
localparam DMA_TASK_LOAD_TASK_END  = 4;  // load task end
localparam DMA_TASK_STORE_TASK_BEG = 5; // store task begin
localparam DMA_TASK_STORE_TASK_END = 7; // store task end

localparam CTRL_CLEAR            = 4'b0000;
localparam CTRL_SHUTDOWN         = 4'b0001;
localparam CTRL_START            = 4'b0010;

/////////////////////////////////////////////////
////// BANK 0 slot table wire declaration ///////
/////////////////////////////////////////////////

reg [BANK0_STATUS_WIDTH-1   : 0]    mainStatus;   ///// 0x4
reg [BANK0_CNT_WIDTH   -1   : 0]    mainCnt;      ///// 0x8
reg [(1 <<BANK0_CNT_WIDTH)-1: 0]    mainTrigger;  
assign slaveReprog = (mainStatus == STATUS_REPROG) ? mainTrigger : 0; //// we want only when it is in stage reprogramming(only 1 cycle)
reg [BANK0_CNT_WIDTH   -1:0]    endCnt; ///// 0xC

reg [GLOB_ADDR_WIDTH   -1:0]    dmaBaseAddr; ///// 0x10
reg [GLOB_ADDR_WIDTH   -1:0]    dfxCtrlAddr; ///// 0x14

reg [BANK0_INTR_WIDTH      -1:0]    intrEna;
reg [BANK0_INTR_WIDTH      -1:0]    intr;
reg [BANK0_ROUNDTRIP_WIDTH -1:0]    roundTrip; /// the round trip counter

reg [DMA_INIT_TASK_CNT -1:0]    dmaInitTask;
////////////////////////////////////////////////
////// restart signal declaration //////////////
////////////////////////////////////////////////
wire finishRound = (mainStatus == STATUS_WAIT4FIN) && (mainCnt == endCnt) && (slave_bank1_out_st_mask == slave_bank1_out_st_intr_mask); ///// the round trip is finished when the mainCnt is equal to endCnt and the slave has finished executing

/////////////////////////////////////////////////
////// BANK 1 slot table wire declaration ///////
/////////////////////////////////////////////////


////// the writing side signal

//////////////  the input pool

wire [BANK1_INDEX_WIDTH    -1:0] bank1_inp_index; // actually it is a wire

wire [BANK1_PROFILE_WIDTH  -1:0] bank1_inp_profile; // it must share with ps and auto inc
wire [BANK1_ST_MSK_WIDTH   -1:0] bank1_inp_st_intr_mask_ack; // mask is usesd only when state is w4fin the interrupt should be set forever, if there is no reset signal
wire [BANK1_ST_MSK_WIDTH   -1:0] bank1_inp_st_intr_mask_abs;

wire bank1_set_fin_src_addr;
wire bank1_set_fin_src_size;
wire bank1_set_fin_des_addr;
wire bank1_set_fin_des_size;
wire bank1_set_fin_status;
wire bank1_set_fin_profile;
wire bank1_set_fin_ld_mask;
wire bank1_set_fin_st_mask;
wire bank1_set_fin_intr_mask_ack;
wire bank1_set_fin_intr_mask_abs;


//////////////  the out pool
wire [BANK1_INDEX_WIDTH    -1:0] bank1_out_index; // actually it is a wire
wire [BANK1_DST_ADDR_WIDTH -1:0] bank1_out_src_addr;      // actually it is a reg
wire [BANK1_DST_SIZE_WIDTH -1:0] bank1_out_src_size;      // actually it is a reg
wire [BANK1_DST_ADDR_WIDTH -1:0] bank1_out_des_addr;
wire [BANK1_DST_SIZE_WIDTH -1:0] bank1_out_des_size;
wire [BANK1_STATUS_WIDTH   -1:0] bank1_out_status  ;      // actually it is a reg
wire [BANK1_PROFILE_WIDTH  -1:0] bank1_out_profile ;      // actually it is a reg
wire [BANK1_LD_MSK_WIDTH   -1:0] bank1_out_ld_mask;
wire [BANK1_ST_MSK_WIDTH   -1:0] bank1_out_st_mask;
wire [BANK1_ST_MSK_WIDTH   -1:0] bank1_out_st_intr_mask;

///////////////////////////////////
//////////// assign bank1       ///
///////////////////////////////////

////////////////////////////////////////
////// reading   ///////////////////////
//////////////////////////////////////// only when the system is in shutdown state, we can read the bank1 slot
assign bank1_out_index = (mainStatus == STATUS_SHUTDOWN) ? ext_bank1_out_index: mainCnt;

assign ext_bank1_out_src_addr      = bank1_out_src_addr;
assign ext_bank1_out_src_size      = bank1_out_src_size;
assign ext_bank1_out_des_addr      = bank1_out_des_addr;
assign ext_bank1_out_des_size      = bank1_out_des_size;
assign ext_bank1_out_status        = bank1_out_status;
assign ext_bank1_out_profile       = bank1_out_profile;
assign ext_bank1_out_ld_mask       = bank1_out_ld_mask;
assign ext_bank1_out_st_mask       = bank1_out_st_mask;
assign ext_bank1_out_st_intr_mask  = bank1_out_st_intr_mask;
assign ext_bank1_out_ready         = ext_bank1_out_req &&  (mainStatus == STATUS_SHUTDOWN);



assign slave_bank1_out_src_addr      =  bank1_out_src_addr;
assign slave_bank1_out_src_size      =  bank1_out_src_size;
assign slave_bank1_out_des_addr      =  bank1_out_des_addr;
assign slave_bank1_out_des_size      =  bank1_out_des_size;
assign slave_bank1_out_status        =  bank1_out_status;
assign slave_bank1_out_profile       =  bank1_out_profile;
assign slave_bank1_out_ld_mask       = bank1_out_ld_mask;
assign slave_bank1_out_st_mask       = bank1_out_st_mask;
assign slave_bank1_out_st_intr_mask  = bank1_out_st_intr_mask;

////////////////////////////////////////
/////// writing ////////////////////////
////////////////////////////////////////
assign bank1_inp_index             = (mainStatus == STATUS_SHUTDOWN) ? ext_bank1_inp_index : mainCnt;

///////////// writing profiler data

assign bank1_inp_profile = ( (mainStatus == STATUS_REPROG      ) | 
                             (mainStatus == STATUS_W4SLAVERESET) |
                             (mainStatus == STATUS_W4SLAVEOP   )) ? (slave_bank1_out_profile + 1) : ext_bank1_inp_profile;

assign bank1_inp_st_intr_mask_ack = mgsFinExec;
assign bank1_inp_st_intr_mask_abs = (mainStatus == STATUS_SHUTDOWN)  ? ext_bank1_inp_st_intr_mask_abs : 0; ///// in case reset mgs 

////////////// writing from external setData
wire ext_bank1_mainActual_set_req = (mainStatus == STATUS_SHUTDOWN) && 
                                    (ext_bank1_set_src_addr | ext_bank1_set_src_size |
                                     ext_bank1_set_des_addr | ext_bank1_set_des_size |
                                     ext_bank1_set_status   | ext_bank1_set_profile  |
                                     ext_bank1_set_ld_mask  | ext_bank1_set_st_mask  |
                                     ext_bank1_set_st_intr_mask_abs);
assign ext_bank1_set_fin_src_addr   = ext_bank1_mainActual_set_req & ext_bank1_set_src_addr;
assign ext_bank1_set_fin_src_size   = ext_bank1_mainActual_set_req & ext_bank1_set_src_size;
assign ext_bank1_set_fin_des_addr   = ext_bank1_mainActual_set_req & ext_bank1_set_des_addr;
assign ext_bank1_set_fin_des_size   = ext_bank1_mainActual_set_req & ext_bank1_set_des_size;
assign ext_bank1_set_fin_status     = ext_bank1_mainActual_set_req & ext_bank1_set_status;
assign ext_bank1_set_fin_profile    = ext_bank1_mainActual_set_req & ext_bank1_set_profile;
assign ext_bank1_set_fin_ld_mask         = ext_bank1_mainActual_set_req & ext_bank1_set_ld_mask;  
assign ext_bank1_set_fin_st_mask         = ext_bank1_mainActual_set_req & ext_bank1_set_st_mask;  
// assign ext_bank1_set_fin_st_intr_mask_ack   = 0;
assign ext_bank1_set_fin_st_intr_mask_abs   = ext_bank1_mainActual_set_req & ext_bank1_set_st_intr_mask_abs;

////////////// writing pool
assign bank1_set_fin_src_addr       = ext_bank1_set_fin_src_addr;
assign bank1_set_fin_src_size       = ext_bank1_set_fin_src_size;
assign bank1_set_fin_des_addr       = ext_bank1_set_fin_des_addr;
assign bank1_set_fin_des_size       = ext_bank1_set_fin_des_size;
assign bank1_set_fin_status         = ext_bank1_set_fin_status;
assign bank1_set_fin_profile        = ext_bank1_set_fin_profile | 
                                    ( ( mainStatus == STATUS_REPROG       ) |
                                      ( mainStatus == STATUS_W4SLAVERESET ) |
                                      ( mainStatus == STATUS_W4SLAVEOP    ));
assign bank1_set_fin_ld_mask        = ext_bank1_set_fin_ld_mask;
assign bank1_set_fin_st_mask        = ext_bank1_set_fin_st_mask;
assign bank1_set_fin_intr_mask_ack  = mainStatus == STATUS_WAIT4FIN; //// this is internal only set
assign bank1_set_fin_intr_mask_abs  = ext_bank1_set_fin_st_intr_mask_abs; //// this is internal reset and external write


///////////////////////////////////////////////
///////// variable setting/getting //////////
///////////////////////////////////////////////

///////////////////////////////////////////////////
//////////// control will be held in state machine
///////////////////////////////////////////////////

////////////////////////////////////
//////////// status setting/////////
////////////////////////////////////
assign ext_bank0_out_status = mainStatus;
////////////////////////////////////
//////////// main counter setting///
////////////////////////////////////
assign ext_bank0_out_mainCnt = mainCnt;
///////////////////////////////////
//////////// end counter setting///
///////////////////////////////////
assign ext_bank0_out_endCnt = endCnt;
always @(posedge clk or negedge reset) begin
    if (~reset) begin
        endCnt  <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        // if the system is in shutdown state, we can set the endCnt
        if (ext_bank0_set_endCnt) begin
            endCnt <= ext_bank0_inp_endCnt;
        end
    end
    // otherwise, we do not allow to set the endCnt register
end

///////////////////////////////////
//////////// dma address setting///
///////////////////////////////////
assign ext_bank0_out_dmaBaseAddr = dmaBaseAddr;
always @(posedge clk or negedge reset)begin
    if (~reset) begin
        dmaBaseAddr <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        if (ext_bank0_set_dmaBaseAddr) begin
            dmaBaseAddr <= ext_bank0_inp_dmaBaseAddr;
        end
    end

end
///////////////////////////////////////////
//////////// dma address setting        ///
///////////////////////////////////////////
assign ext_bank0_out_dfxCtrlAddr = dfxCtrlAddr;
always @(posedge clk or negedge reset)begin
    if (~reset) begin
        dfxCtrlAddr <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        if (ext_bank0_set_dfxCtrlAddr) begin
            dfxCtrlAddr <= ext_bank0_inp_dfxCtrlAddr;
        end
    end 
end

///////////////////////////////////////////
//////////// interrupt  enable          ///
///////////////////////////////////////////
assign ext_bank0_out_intrEna = intrEna;
always @(posedge clk or negedge reset) begin
    if (~reset) begin
        intrEna <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        if (ext_bank0_set_intrEna) begin
            intrEna <= ext_bank0_inp_intrEna; // set the interrupt enable signal
        end
    end
end

///////////////////////////////////////////
//////////// interrupt signal           ///
///////////////////////////////////////////
assign ext_bank0_out_intr = intr;
always @(posedge clk or negedge reset) begin
    if (~reset) begin
        intr <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        if ((ext_bank0_set_intr && ext_bank0_inp_intr) || hw_intr_clear) begin
            intr <= 0; // reset the interrupt signal
        end
    end else if (finishRound && intrEna)begin 
            intr <= 1;
    end
end

///////////////////////////////////////////
//////////// round trip                 ///
///////////////////////////////////////////
assign ext_bank0_out_roundTrip = roundTrip;
always @(posedge clk or negedge reset) begin
    if (~reset) begin
        roundTrip <= 0;
    end else if (mainStatus == STATUS_SHUTDOWN) begin
        if (ext_bank0_set_roundTrip) begin
            roundTrip <= ext_bank0_inp_roundTrip; // set the round trip counter
        end
    end else if (finishRound) begin
        roundTrip <= roundTrip + 1; // increment the round trip counter
    end
end


/////////////////////////////////////////////////////
//////////// dma and mgs stream initialization //////
/////////////////////////////////////////////////////
assign slaveMgsLoadReset  = (mainStatus == STATUS_CLEAR_MGS)      ? bank1_out_ld_mask : 0; ///// the magic sequencer will load the data from the bank1 slot
assign slaveMgsStoreReset = (mainStatus == STATUS_CLEAR_MGS)      ? bank1_out_st_mask : 0; ///// the magic sequencer will load the data from the bank1 slot
assign slaveMgsLoadInit   = (mainStatus == STATUS_INITIALIZE_MGS) ? bank1_out_ld_mask : 0; ///// the magic sequencer will load the data from the bank1 slot
assign slaveMgsStoreInit  = (mainStatus == STATUS_INITIALIZE_MGS) ? bank1_out_st_mask : 0; ///// the magic sequencer will store the data to the bank1 slot
assign slaveInit[DMA_INIT_TASK_CNT-1: 0] = dmaInitTask[DMA_INIT_TASK_CNT-1: 0];


//////////////////////////////////////////////
///////// CONTROL SYSTEM  /////////////////////
///////////////////////////////////////////////

always @(posedge clk or negedge reset ) begin
    
    ///// case the system is commanded by the outsider
    if (~reset) begin
        mainStatus <= STATUS_SHUTDOWN;
        mainCnt    <= 0;
        mainTrigger <= 0;
        dmaInitTask <= 0;
    end else if (ext_bank0_set_control) begin
        case (ext_bank0_inp_control)
            CTRL_CLEAR: begin
                mainStatus  <= STATUS_SHUTDOWN;
                mainCnt     <= 0;
                mainTrigger <= 0;
                dmaInitTask <= 0;
            end
            CTRL_SHUTDOWN: begin
                mainStatus <= STATUS_SHUTDOWN;
            end
            CTRL_START: begin
                if ( (mainStatus == STATUS_SHUTDOWN) && (~intr) ) begin
                    mainStatus <= STATUS_REPROG;
                    mainCnt    <= 0; // reset the counter
                    mainTrigger<= 1;
                end
            end
            default: begin
                // do nothing, just keep the current status
            end
        endcase
    end else begin
        
        case (mainStatus)
            STATUS_SHUTDOWN: begin
                if (hw_ctrl_start && (~intr)) begin
                    mainStatus <= STATUS_REPROG; // go to reprogramming state
                    mainCnt    <= 0; // reset the counter
                    mainTrigger <= 1; // set the trigger to 1
                end
            end
            STATUS_REPROG: begin
                // do nothing, just keep the current status
                // we can trigger the slave to do something
                
                // if (slaveReprogAccept) begin
                //     mainStatus <= STATUS_INITIALIZING; // go to initializing state
                //     dmaInitTask <= 1; //// intializee the init task 
                // end
                mainStatus  <= STATUS_W4SLAVERESET;
            end
            STATUS_W4SLAVERESET: begin
                ///// wait4 reset occur
                if (~nslaveReset)begin
                    mainStatus <= STATUS_W4SLAVEOP;
                end

            end
            STATUS_W4SLAVEOP: begin
                if (nslaveReset) begin
                    mainStatus  <= STATUS_CLEAR_MGS;
                end
            end
            STATUS_CLEAR_MGS: begin
                mainStatus <= STATUS_INITIALIZE_MGS;
            end
            STATUS_INITIALIZE_MGS: begin
                    mainStatus  <= STATUS_INITIALIZE_DMA;
                    dmaInitTask <= 1;
            end
            STATUS_INITIALIZE_DMA: begin
                // do nothing, just keep the current status
                // we can trigger the slave to do something
                if (slaveFinInit[DMA_TASK_RESET_INTR_END]) begin /// slaveFinInit is one cycle
                    if(bank1_out_ld_mask[0])begin
                        dmaInitTask <= 1 << DMA_TASK_LOAD_TASK_BEG; /// shift to the next task
                        mainStatus  <= STATUS_SET_DMA_LOAD;
                    end else if(bank1_out_st_mask[0]) begin
                        dmaInitTask <= 1 << DMA_TASK_STORE_TASK_BEG; /// shift to the next task
                        mainStatus  <= STATUS_SET_DMA_STORE;
                    end else begin
                        dmaInitTask <= 0; /// no task to do
                        mainStatus  <= STATUS_TRIGGERING; // go to load state
                    end
                end else if (slaveFinInit != 0) begin
                    dmaInitTask <= dmaInitTask << 1; /// shift to the next task
                end
            end
            STATUS_SET_DMA_LOAD: begin
                // do nothing, just keep the current status
                // we can trigger the slave to do something
                if (slaveFinInit[DMA_TASK_LOAD_TASK_END]) begin /// slaveFinInit is one cycle
                    if(bank1_out_st_mask[0]) begin
                        dmaInitTask <= 1 << DMA_TASK_STORE_TASK_BEG; /// shift to the next task
                        mainStatus <= STATUS_SET_DMA_STORE; // go to store state
                    end else begin
                        dmaInitTask <= 0;
                        mainStatus  <= STATUS_TRIGGERING; // go to triggering state
                    end
                end else if (slaveFinInit != 0) begin
                    dmaInitTask <= dmaInitTask << 1; /// shift to the next task
                end

            end
            STATUS_SET_DMA_STORE: begin
                // do nothing, just keep the current status
                // we can trigger the slave to do something
                if (slaveFinInit[DMA_TASK_STORE_TASK_END]) begin /// slaveFinInit is one cycle
                    dmaInitTask <= 0; /// no task to do
                    mainStatus <= STATUS_TRIGGERING; // go to triggering state
                end else if (slaveFinInit != 0) begin
                    dmaInitTask <= dmaInitTask << 1; /// shift to the next task
                end 
            end
            STATUS_TRIGGERING: begin
                mainStatus <= STATUS_WAIT4FIN;
            end
            STATUS_WAIT4FIN: begin
                if(slave_bank1_out_st_mask == slave_bank1_out_st_intr_mask) begin
                    // the slave has finished executing, we can go to the next step
                    if (mainCnt < endCnt) begin
                        mainCnt <= mainCnt + 1; // increment the counter
                        mainTrigger <= (mainTrigger << 1);
                        mainStatus <= STATUS_REPROG; // go back to initializing state
                    end else begin
                        mainCnt     <= 0; // reset the counter
                        mainTrigger <= 0;
                        mainStatus <= STATUS_SHUTDOWN; // go back to shutdown state
                    end
                end
            end
            default: begin
                // do nothing, just keep the current status
            end
        endcase
    end
    
end

///////////////////////////////////////////////
///////// BANK1 slot table ///////////////////
///////////////////////////////////////////////

SlotArr #(
.INDEX_WIDTH    (BANK1_INDEX_WIDTH),
.SRC_ADDR_WIDTH (BANK1_SRC_ADDR_WIDTH),
.SRC_SIZE_WIDTH (BANK1_SRC_SIZE_WIDTH),
.DST_ADDR_WIDTH (BANK1_DST_ADDR_WIDTH),
.DST_SIZE_WIDTH (BANK1_DST_SIZE_WIDTH),
.STATUS_WIDTH   (BANK1_STATUS_WIDTH),
.PROFILE_WIDTH  (BANK1_PROFILE_WIDTH),
.LD_MSK_WIDTH   (BANK1_LD_MSK_WIDTH),
.ST_MSK_WIDTH   (BANK1_ST_MSK_WIDTH)
) dayta (
    .clk(clk),
    .reset(reset),
    // Declare an array of slots
    .inp_index              (bank1_inp_index),
    .inp_src_addr           (ext_bank1_inp_src_addr),
    .inp_src_size           (ext_bank1_inp_src_size),
    .inp_des_addr           (ext_bank1_inp_des_addr),
    .inp_des_size           (ext_bank1_inp_des_size),
    .inp_status             (ext_bank1_inp_status),
    .inp_profile            (bank1_inp_profile), // it must share with ps and auto inc
    .inp_ld_mask            (ext_bank1_inp_ld_mask),
    .inp_st_mask            (ext_bank1_inp_st_mask),
    .inp_st_intr_mask_ack   (bank1_inp_st_intr_mask_ack),
    .inp_st_intr_mask_abs   (bank1_inp_st_intr_mask_abs),

    .set_src_addr         (bank1_set_fin_src_addr),
    .set_src_size         (bank1_set_fin_src_size),
    .set_des_addr         (bank1_set_fin_des_addr),
    .set_des_size         (bank1_set_fin_des_size),
    .set_status           (bank1_set_fin_status  ),
    .set_profile          (bank1_set_fin_profile ),
    .set_ld_mask          (bank1_set_fin_ld_mask),
    .set_st_mask          (bank1_set_fin_st_mask),
    .set_st_intr_mask_ack (bank1_set_fin_intr_mask_ack),
    .set_st_intr_mask_abs (bank1_set_fin_intr_mask_abs),

    // Output ports0
    .out_index        (bank1_out_index),
    .out_src_addr     (bank1_out_src_addr),      // actually it is a wire
    .out_src_size     (bank1_out_src_size),      // actually it is a wire
    .out_des_addr     (bank1_out_des_addr),      // actually it is a wire
    .out_des_size     (bank1_out_des_size),      // actually it is a wire
    .out_status       (bank1_out_status) ,      // actually it is a wire
    .out_profile      (bank1_out_profile),       // actually it is a wire
    .out_ld_mask      (bank1_out_ld_mask),
    .out_st_mask      (bank1_out_st_mask),
    .out_st_intr_mask (bank1_out_st_intr_mask)    
);

endmodule
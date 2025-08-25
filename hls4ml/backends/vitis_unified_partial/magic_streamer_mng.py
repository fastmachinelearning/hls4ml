class MgsConMeta:
    ####
    def __init__(self, io_idx, tensor):
        self.io_idx            = io_idx
        self.mgs_idx           = -1
        self.mgs_wrap_width    = 32
        self.mgs_row_idx_width = 10


class MagicBufferMeta:
    def __init__(self, data_width, row_idx_width, mgs_idx):
        self.data_width    = data_width
        self.row_idx_width = row_idx_width
        self.mgs_idx       = mgs_idx

    def upgrade_mgs_to_support(self, mgs_con_meta):
        if self.row_idx_width < mgs_con_meta.mgs_row_idx_width:
            self.row_idx_width = mgs_con_meta.mgs_row_idx_width


    def is_data_width_match(self, check_width):
        return self.data_width == check_width

    #### the shared class between subgraph and host graph

class MgsConGraph:
    #### in suppose to be pool of connection for each sub graph

    def __init__(self, gid, input_node_links, amt_graph, mgs_model):  ### gid = graph id
        self.gid = gid
        self.input_cons       = []  #### the index of the list supposed to be the index of the input port as well
        self.output_cons      = []
        self.input_node_links = input_node_links
        self.amt_graph        = amt_graph
        self.mgs_model    = mgs_model

    def start_convert_graph(self, graph):
        #### do the output first because we need to not free the src first (src des cannot be the same buffer)
        out_var = graph.get_output_variables()
        for out_idx, out in enumerate(out_var):
            mgs_con_meta = MgsConMeta(out_idx, out)
            self.add_output_con(mgs_con_meta)

        #### do the input
        in_var = graph.get_input_variables()
        for in_idx, inp in enumerate(in_var):
            mgs_con_meta = MgsConMeta(in_idx, inp)
            self.add_input_con(mgs_con_meta)

    def is_last_graph(self):
        return (self.gid + 1) == self.amt_graph

    def add_input_con(self, mgs_con_meta):

        src_gid, src_out_idx = self.input_node_links[self.gid][mgs_con_meta.io_idx]

        #### it load from dma
        if src_gid == -1:
            src_mgs_idx = -1  ##### it means dma
        else:
            src_mgs_con_meta = self.mgs_model.get_mgs_idx(src_gid, src_out_idx)
            mgs_con_meta.mgs_idx = src_mgs_con_meta
            self.mgs_model.move_buffer_to_free_list(src_mgs_con_meta)

        self.input_cons.append(mgs_con_meta)

    def add_output_con(self, mgs_con_meta):

        if self.is_last_graph():
            self.output_cons.append(mgs_con_meta)
            return

        ###### we check the
        stream_buffer_idx = self.mgs_model.get_existing_possible_mgs_buffer(mgs_con_meta)

        if stream_buffer_idx is None:
            stream_buffer_idx = self.mgs_model.allocate_mgs_buffer(mgs_con_meta)

        ##### upgrade the magic stream buffer to match size of it is lower
        self.mgs_model.upgrade_mgs_to_support(mgs_con_meta, stream_buffer_idx)

        self.mgs_model.move_buffer_to_using_list(stream_buffer_idx)
        mgs_con_meta.mgs_idx = stream_buffer_idx


        self.output_cons.append(mgs_con_meta)

class MgsModel:
    def __init__(self, multigraph):
        self.multigraph = multigraph
        self.con_graphs      = []
        self.mgs_buffer_meta = []  #### index of the system supposed to be magic streamer and its port id

        self.mgs_buffer_holding = []
        self.mgs_buffer_empty   = []

    ##############################################
    ############ start get the data    ###########
    ##############################################


    def start_convert_model(self):
        for gid, sub_graph in enumerate(self.multigraph.graphs):
            #### initialize the MgsConGraph
            input_node_link = self.multigraph.input_node_links
            amt_graph       = len(self.multigraph.graphs)
            mgs_con_graph = MgsConGraph(gid, input_node_link, amt_graph, self)
            mgs_con_graph.start_convert_graph(sub_graph)
            #### start fill the metadata
            self.add_mgs_con_graph(mgs_con_graph)

    def add_mgs_con_graph(self, mgs_con_graph):
        self.con_graphs.append(mgs_con_graph)


    ### for outsider used
    def get_mgs_idx_src(self, gid, inputIdx):
        if gid not in range(0, self.multigraph.amt_graph):
            raise Exception(
                "get_mgs_idx_src: gid {gid} is out of bound. The amount of graph is {amt_graph}".format(
                    gid=gid, amt_graph=self.multigraph.amt_graph
                )
            )
        if inputIdx not in range(0, len(self.con_graphs[gid].input_cons)):
            raise Exception(
                "get_mgs_idx_src: inputIdx {inputIdx} is out of bound. The amount of input is {input_num}".format(
                    inputIdx=inputIdx, input_num=len(self.con_graphs[gid].input_cons)
                )
            )
        return self.con_graphs[gid].input_cons[inputIdx].mgs_idx
    ### for outsider usd
    def get_mgs_idx_dst(self, gid, outputIdx):
        if gid not in range(0, self.multigraph.amt_graph):
            raise Exception(
                "get_mgs_idx_dst: gid {gid} is out of bound. The amount of graph is {amt_graph}".format(
                    gid=gid, amt_graph=self.multigraph.amt_graph
                )
            )
        if outputIdx not in range(0, len(self.con_graphs[gid].output_cons)):
            raise Exception(
                "get_mgs_idx_dst: outputIdx {outputIdx} is out of bound. The amount of output is {output_num}".format(
                    outputIdx=outputIdx, output_num=len(self.con_graphs[gid].output_cons)
                )
            )
        return self.con_graphs[gid].output_cons[outputIdx].mgs_idx

    ##############################################
    ############ magic streamer buffer ###########
    ##############################################

    ##### used by the vitis uniifed partial backend writer and MgsConGraph
    def get_mgs_idx(self, gid, outputIdx):
        if gid < 0:
            return -1
        mgs_con_meta = self.con_graphs[gid].output_cons[outputIdx]
        return mgs_con_meta.mgs_idx

    def upgrade_mgs_to_support(self, mgs_con_meta, mgsIdx):
        if mgsIdx >= len(self.mgs_buffer_meta) or mgsIdx < 0:
            raise Exception("upgrade magic streamer with Idx {mgsIdx} is out of bound.")
        self.mgs_buffer_meta[mgsIdx].upgrade_mgs_to_support(mgs_con_meta)

    def allocate_mgs_buffer(self, mgs_con_meta):
        newStreamBuffer = MagicBufferMeta(mgs_con_meta.mgs_wrap_width, mgs_con_meta.mgs_row_idx_width, len(self.mgs_buffer_meta))
        mgs_idx = len(self.mgs_buffer_meta)
        self.mgs_buffer_meta.append(newStreamBuffer)
        return mgs_idx

    def get_existing_possible_mgs_buffer(self, mgs_con_meta):
        ##### filter the match buffer from exis
        matched_buffer = list(filter(
            lambda mgs: mgs.is_data_width_match(mgs_con_meta.mgs_wrap_width),
            self.mgs_buffer_meta ))

        highest_possible_buffer = sorted(matched_buffer, key=lambda x: x.row_idx_width, reverse=True)

        return None if len(highest_possible_buffer) == 0 else highest_possible_buffer[0].mgs_idx

    def move_buffer_to_using_list(self, mgs_idx):
        ###### delete from free list first
        self.mgs_buffer_empty = list(filter(lambda x: x.mgs_idx != mgs_idx, self.mgs_buffer_empty))
        ###### add to holding list
        self.mgs_buffer_holding.append(self.mgs_buffer_meta[mgs_idx])

    def move_buffer_to_free_list(self, mgs_idx):
        ###### delete from holding list first
        self.mgs_buffer_holding = list(filter(lambda x: x.mgs_idx != mgs_idx, self.mgs_buffer_holding))
        ###### add to free list
        self.mgs_buffer_empty.append(self.mgs_buffer_meta[mgs_idx])
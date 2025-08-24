

#### the shared class between subgraph and host graph

class MgsConMeta:
    ####
    def __init__(self, io_idx, input_node, mgs_wrap_width, mgs_row_idx_width):
        self.io_idx         = io_idx
        self.input_node     = input_node
        self.mgs_idx        = -1
        self.mgs_wrap_width = mgs_wrap_width
        self.mgs_row_idx_width = mgs_row_idx_width

class MagicStreamerMeta:
    def __init__(self, data_width, row_idx_width, mgs_idx):
        self.data_width    = data_width
        self.row_idx_width = row_idx_width
        self.mgs_idx       = mgs_idx


class MagicStreamerManager:

    ###
    ### [ (inputIdx, mgsNumber, mgsNumber) ]
    ### []

    def __init__(self, multiModel):

        self.magicStreamer_meta_list      = []     ##### (data_width, row_idx_width, mgs_idx)
        self.input_connection_result_list = [[] for _ in range(len(multiModel.graphs))]


    def get_MgsConMeta_from_graph(self, graph, input_node,io_idx, is_input):

        ####### get its tensor variable
        tensor_var = graph.get_input_variables()[io_idx] if is_input else graph.get_output_variables()[io_idx]
        ####### fix this two variable
        mgs_bit_width = 32
        mgs_row_width = 1024
        mgs_con_meta = MgsConMeta(io_idx, input_node, mgs_bit_width, mgs_row_width)
        return mgs_con_meta

    def complete_the_connection(self, mgsCon: MgsConMeta, multiModel):
        ###[ [ (inputNode, srcNode, srcNodeOutputIdx), .....], [ ..], [ ..] ]
        node_links = multiModel.input_node_links

        inspect_input_node = mgsCon.input_node

        for input_node, src_node, src_node_output_idx in node_links:
            if input_node == inspect_input_node:
                #### find where is so


    def select_mgs_number_for_port(self, inspect_connection: MgsConMeta, free_list: list[MagicStreamerMeta]):

        matched_dw_mgs = list(filter(lambda x: x.data_width == inspect_connection.mgs_wrap_width, free_list))
        sorted_mgs = sorted(matched_dw_mgs, key=lambda x: x.row_idx_width, reverse=True)
        #### get the MagicStreamerMeta that have the maximum size
        if len(sorted_mgs) == 0:
            return None
        else:
            return sorted_mgs[0]


    def build_all_meta_data(self, multiModel):

        next_mgs_number = 0
        ready_to_used_mgs = [] ### list of mgs number
        using_mgs_list    = [] ### list of mgs number


        #### loop to all subgraph
        for gid, subGraph in enumerate(multiModel.graphs):

            ######## we inspect the output
            connection_out_meta_list = [ self.get_MgsConMeta_from_graph(subGraph, subGraph.graph[io_name], io_idx, False)
                                         for io_idx, io_name in enumerate(subGraph.outputs)]
            connection_out_meta_sorted_rs = sorted(connection_out_meta_list, key=lambda x: x.mgs_row_idx_width, reverse=True)

            ##### try to allocate the magic streamer that matched the most match
            for connection in connection_out_meta_sorted_rs:
                ##### check the available magic streamer for each connection
                selected_mgs = self.select_mgs_number_for_port(connection, ready_to_used_mgs)
                if selected_mgs is None:
                    ##### not found create the magic streamer
                    using_mgs_list.append(MagicStreamerMeta(connection.mgs_wrap_width, connection.mgs_row_idx_width, next_mgs_number))
                else:
                    ##### found get the increase size of row Idx if it is need
                    idx_in_ready_list = ready_to_used_mgs.index(selected_mgs)
                    ready_to_used_mgs.pop(idx_in_ready_list) #### remove it from ready list
                    using_mgs_list.append(selected_mgs)      #### append it into used list

            ######## we inspect the input
            connection_in_meta_list = [self.get_MgsConMeta_from_graph(subGraph, subGraph.graph[io_name], io_idx, True)
                                        for io_idx, io_name in enumerate(subGraph.inputs)]
            for connection in connection_in_meta_list:





        ########## finalize the magic streamer metadata
        self.magic_streamer_meta_list = sorted(ready_to_used_mgs, key=lambda x: x.mgs_idx)
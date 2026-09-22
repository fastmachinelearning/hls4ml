from hls4ml.backends.vitis.passes import fifo_depth_optimization as vitis_fifo_opt


class FifoDepthOptimizationPost(vitis_fifo_opt.FifoDepthOptimizationPost):
    def get_hls_project_path(self, model):
        return model.config.backend.writer.get_vitis_hls_exec_dir(model) + '/hls'

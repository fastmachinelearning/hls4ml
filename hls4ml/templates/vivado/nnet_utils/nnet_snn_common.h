#ifndef NNET_SNN_COMMON_H_
#define NNET_SNN_COMMON_H_

namespace nnet {

enum class snn_reset_mode { subtract, zero };
enum class snn_decision_rule { argmax_spike_count, first_to_threshold, threshold_then_argmax, binary_logit, argmax_membrane };
enum class snn_readout_mode { spike, membrane };

} // namespace nnet

#endif

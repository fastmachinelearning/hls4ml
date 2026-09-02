import json

import numpy as np

from hls4ml.utils.serialization import _deserialize_array_attrs, _serialize_array_attrs


def test_serialize_array_attrs_in_sequences(tmp_path):
    """Arrays nested inside lists/tuples (e.g., mask_kbi of FixedPointQuantizer) should be serialized. See #1526."""
    model_arch = {
        'layer': {
            'state': {
                'attributes': {
                    'name': 'layer',
                    'mask_kbi': (
                        np.array([[0, 1, 0]], dtype=np.int16),
                        np.array([[1, 2, 3]], dtype=np.int16),
                        np.array([[4, 5, 6]], dtype=np.int16),
                    ),
                    'nested_list': [[np.array([1.5, 2.5])], 'foo', np.int32(7)],
                }
            }
        }
    }

    for layer_name, layer_dict in model_arch.items():
        _serialize_array_attrs(layer_dict, layer_name, tmp_path)

    attrs = model_arch['layer']['state']['attributes']
    assert attrs['mask_kbi'] == (
        '@ndarray:layer_mask_kbi_0.npy',
        '@ndarray:layer_mask_kbi_1.npy',
        '@ndarray:layer_mask_kbi_2.npy',
    )
    assert attrs['nested_list'] == [['@ndarray:layer_nested_list_0_0.npy'], 'foo', 7]

    # The resulting architecture must be JSON serializable
    arch_json = json.dumps(model_arch)

    # Round-trip: arrays inside (JSON-decoded) lists should be restored on deserialization
    restored_arch = json.loads(arch_json)
    _deserialize_array_attrs(tmp_path, restored_arch)

    restored_attrs = restored_arch['layer']['state']['attributes']
    k, b, i = restored_attrs['mask_kbi']
    np.testing.assert_array_equal(k, np.array([[0, 1, 0]], dtype=np.int16))
    np.testing.assert_array_equal(b, np.array([[1, 2, 3]], dtype=np.int16))
    np.testing.assert_array_equal(i, np.array([[4, 5, 6]], dtype=np.int16))
    np.testing.assert_array_equal(restored_attrs['nested_list'][0][0], np.array([1.5, 2.5]))
    assert restored_attrs['nested_list'][1:] == ['foo', 7]

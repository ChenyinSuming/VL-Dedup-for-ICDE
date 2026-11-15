import numpy as np
from PIL import Image
from scripts.build_edit_stress import parameters, transform

def test_edits_use_fixed_parameters_and_center_crop():
    image = Image.new('RGB', (100, 80), (20, 80, 120))
    params = parameters('mild', 12)
    (first, box) = transform(image, params)
    (second, repeated) = transform(image, parameters('mild', 12))
    assert box == repeated == (6, 5, 94, 75)
    assert first.size == (88, 70)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert params['quality'] == 70 and params['crop_fraction'] == 0.12

from mnistclassifier.activation_functions import sigmoid


def test_sigmoid_function_returns_correct_value_on_upper_bound():
    assert sigmoid(37) == 1.0

def test_sigmoid_function_returns_correct_value_on_zero():
    assert sigmoid(0) == 0.5

def test_sigmoid_function_returns_correct_value_on_lower_bound():
    assert sigmoid(-10) < 0.01

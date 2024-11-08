from arc.external import openai


def test_complete():
    output = openai.complete("Say hello.")
    assert isinstance(output, str)

from llm_wrappers.token_counter import TokenCounter


def test_token_counter_basic():
    tc = TokenCounter()
    n = tc.count("Hello world")
    assert isinstance(n, int) and n >= 1

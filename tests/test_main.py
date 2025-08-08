from app.main import greet


def test_greet():
    """Test greet function output"""
    assert greet("World") == "Hello World!"

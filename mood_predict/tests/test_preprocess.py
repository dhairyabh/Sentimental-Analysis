import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import pre_process_text

def test_pre_process_text():
    # Test case 1: Normal text
    input_text = "I am FEELING happy!"
    expected = "feel happy"
    assert pre_process_text(input_text) == expected
    
    # Test case 2: Text with punctuation and stop words
    input_text = "This is a test, with some dots..."
    expected = "test dot"
    assert pre_process_text(input_text) == expected
    
    # Test case 3: Empty string
    input_text = ""
    expected = ""
    assert pre_process_text(input_text) == expected
    
    # Test case 4: Non-string input
    input_text = None
    expected = ""
    assert pre_process_text(input_text) == expected

    print("All preprocessing tests passed!")

if __name__ == "__main__":
    test_pre_process_text()

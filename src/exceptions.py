import sys
import os
from src.logger import logging

def get_detailed_error(error, error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    line_number = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"
    return error_message

# Create a class for exception handling
class CustomException(Exception):
    
    def __init__(self, error_msg, error_details:sys):
        super().__init__(error_msg)
        self.error_msg= get_detailed_error(error_msg, error_details)
        
    def __str__(self):
        return self.error_msg
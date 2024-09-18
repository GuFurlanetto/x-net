import json
import csv


# JSON Functions
def read_json(file_path):
    """
    Read a JSON file and return the data as a dictionary.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    dict: Data contained in the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path, indent=4):
    """
    Write data to a JSON file with pretty printing.

    Parameters:
    data (dict): Data to be written to the JSON file.
    file_path (str): Path to the JSON file.
    indent (int, optional): Number of spaces for indentation. Default is 4.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


# CSV Functions
def read_csv(file_path):
    """
    Read a CSV file and return the data as a list of dictionaries.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    list of dict: Data contained in the CSV file, where each row is a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def write_csv(data, file_path, fieldnames):
    """
    Write a list of dictionaries to a CSV file.

    Parameters:
    data (list of dict): Data to be written to the CSV file, where each dictionary represents a row.
    file_path (str): Path to the CSV file.
    fieldnames (list of str): List of field names for the CSV file header.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Text Functions
def read_text(file_path):
    """
    Read a text file and return its content as a string.

    Parameters:
    file_path (str): Path to the text file.

    Returns:
    str: Content of the text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_text(data, file_path):
    """
    Write a string to a text file.

    Parameters:
    data (str): String to be written to the text file.
    file_path (str): Path to the text file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


# Append to Text File
def append_text(data, file_path):
    """
    Append a string to a text file.

    Parameters:
    data (str): String to be appended to the text file.
    file_path (str): Path to the text file.
    """
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(data + "\n")

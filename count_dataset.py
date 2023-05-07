import os

def count_files_by_extension(path, extension, e2):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension) or file.endswith(e2):
                count += 1
    return count

# Example usage
path = "images"
extension = ".jpeg"
e2 = ".jpg"
file_count = count_files_by_extension(path, extension, e2)
print(f"Number of {extension} files in directory tree: {file_count}")

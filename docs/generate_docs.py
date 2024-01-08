import os

src_path = './src'
docs_path = './docs'

for folder in os.listdir(src_path):
    folder_path = os.path.join(src_path, folder)
    if os.path.isdir(folder_path):
        rst_file_path = os.path.join(docs_path, f'{folder}.rst')
        with open(rst_file_path, 'w') as rst_file:
            rst_file.write(f"{folder}\n")
            rst_file.write(f"{'=' * len(folder)}\n\n")
            for file in os.listdir(folder_path):
                if file.endswith('.py'):
                    module = file[:-3]  # Remove .py extension
                    full_module_path = f"{folder}.{module}"
                    rst_file.write(f".. automodule:: {full_module_path}\n")
                    rst_file.write("   :members:\n")
                    rst_file.write("   :undoc-members:\n")
                    rst_file.write("   :show-inheritance:\n\n")

import re
import sys

# Read the file
try:
    with open('web_app/combined_script.py', 'r') as f:
        content = f.read()
    
    # Create a backup
    with open('web_app/combined_script_backup.py', 'w') as f:
        f.write(content)
    
    # Find all JavaScript blocks and wrap them in raw Python triple quotes
    # First, find the start of the script section
    start_index = content.find('            <script>')
    if start_index != -1:
        # Find the closing script tag
        end_index = content.find('            </script>', start_index)
        if end_index != -1:
            # Extract the JavaScript code
            js_code = content[start_index:end_index]
            
            # Wrap the JavaScript code in raw triple quotes
            modified_js = content[start_index:start_index+20] + "'''" + content[start_index+20:end_index] + "'''" + content[end_index:end_index+21]
            
            # Replace the original JavaScript
            content = content[:start_index] + modified_js + content[end_index+21:]
    
    # Also handle the second script section
    start_index = content.find('                <script>')
    if start_index != -1:
        # Find the closing script tag
        end_index = content.find('                </script>', start_index)
        if end_index != -1:
            # Extract the JavaScript code
            js_code = content[start_index:end_index]
            
            # Wrap the JavaScript code in raw triple quotes
            modified_js = content[start_index:start_index+24] + "'''" + content[start_index+24:end_index] + "'''" + content[end_index:end_index+25]
            
            # Replace the original JavaScript
            content = content[:start_index] + modified_js + content[end_index+25:]

    # Fix any template literals with the proper Python f-string syntax
    content = re.sub(r'`\${([^}]+)}%`', r'`${{\\1}}%`', content)
    content = re.sub(r'`\${([^}]+)}`', r'`${{\\1}}`', content)

    # Write back the modified content
    with open('web_app/combined_script.py', 'w') as f:
        f.write(content)
    
    print('Done fixing JavaScript blocks in Python strings')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1) 
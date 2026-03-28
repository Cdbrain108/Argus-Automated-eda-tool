import re
import os

files = ['home.py', 'auth.py']
for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    def replacer(match):
        inner = match.group(1)
        # remove blank lines
        clean_inner = re.sub(r'\n[ \t]*\n', '\n', inner)
        return 'st.markdown("""' + clean_inner + '""", unsafe_allow_html=True)'

    content = re.sub(r'st\.markdown\(\"\"\"(.*?)\"\"\",\s*unsafe_allow_html=True\)', replacer, content, flags=re.DOTALL)
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
print('Fixed blank lines in ' + str(files))

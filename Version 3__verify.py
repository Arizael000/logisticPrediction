import json
with open('3.EDA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Cell 89:', repr(''.join(nb['cells'][89]['source'])))
print('Cell 90:', repr(''.join(nb['cells'][90]['source'])))
print()
print('All creation+log combos:')
pairs = [(90,91,'lacteos'),(94,95,'descartables'),(98,99,'carnes'),
         (103,104,'bebidas'),(108,109,'enlatados'),(112,113,'embutidos'),
         (116,117,'congelados')]
for ci, li, name in pairs:
    csrc = ''.join(nb['cells'][ci]['source'])
    lsrc = ''.join(nb['cells'][li]['source'])
    print(f'  {name}: cre={repr(csrc[:60])} | log={repr(lsrc[:60])}')

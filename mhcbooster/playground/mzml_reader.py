import re
import time

import xml.etree.ElementTree as ET

# 7381/s
# start_time = time.time()
# for event, elem in ET.iterparse('/mnt/d/data/JY_1_10_25M/msconvert/JY_Class1_1M_DDA_60min_Slot1-10_1_541.mzML', events=('end',)):
#     if elem.tag.endswith('spectrum'):
#         index = elem.attrib.get('index')
#         spectrum_id = elem.attrib.get('id')
#         print(f"index: {index}, id: {spectrum_id}")
#         elem.clear()  # 清理释放内存
#
#         if time.time() - start_time > 10:
#             break
# print('debug')

# 5837/s
# pattern = re.compile(r'<spectrum[^>]*index="([^"]+)"[^>]*id="([^"]+)"')
# start_time = time.time()
# with open('/mnt/d/data/JY_1_10_25M/msconvert/JY_Class1_1M_DDA_60min_Slot1-10_1_541.mzML', 'r', encoding='utf-8') as f:
#     for line in f:
#         if not line.lstrip().startswith('<spectrum'):
#             continue
#         match = pattern.search(line)
#         if match:
#             index, spectrum_id = match.groups()
#             print(f"index={index}, id={spectrum_id}")
#         if time.time() - start_time > 10:
#             break

# 6090/s
# from pyteomics import mzml, mgf
# start_time = time.time()
# for l in mzml.read('/mnt/d/data/JY_1_10_25M/msconvert/JY_Class1_1M_DDA_60min_Slot1-10_1_541.mzML', decode_binary=False):
#     print(l['index'], l['id'])
#     if time.time() - start_time > 10:
#         break

# from pyteomics import mzml
# start_time = time.time()
# for l in mzml.read('/mnt/f/MIDIA_dilution/mzml/M2716_calibrated.mzML'):
#     if l['index'] % 100 == 0:
#         print(l['index'], l['id'])
#     if time.time() - start_time > 10:
#         break

from pyopenms import MzMLFile, MSExperiment
start_time = time.time()
exp = MSExperiment()
mzml_file = MzMLFile().load('/mnt/f/MIDIA_dilution/mzml/M2716_calibrated.mzML', exp)
print(time.time() - start_time)
for i, spectrum in enumerate(mzml_file):
    if i % 100 == 0:
        print(i, spectrum.getRT())
    if time.time() - start_time > 10:
        break
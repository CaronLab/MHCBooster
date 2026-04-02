from typing import List
from mhcnames import normalize_allele_name


def prepare_class_I_alleles(alleles: List[str], avail_alleles: List[str]):
    # avail_allele format: HLA-B*27:40
    avail_alleles.sort()
    prepared_alleles = []
    for allele in alleles:
        try:
            allele = normalize_allele_name(allele)
            if allele in avail_alleles:
                prepared_alleles.append(allele)
            else:
                allele = allele.split(':')[0]
                for j in range(len(avail_alleles)):
                    if avail_alleles[j].startswith(allele):
                        prepared_alleles.append(avail_alleles[j])
                        break
        except ValueError:
            print(f'ERROR: Allele {allele} not supported.')
            continue

    return list(set(prepared_alleles))


def prepare_class_II_alleles(alleles: List[str], avail_alleles: List[str]):
    alleles = [allele.replace('HLA-', '') for allele in alleles]
    avail_alleles.sort()
    allele_keywords = ['DRB', 'DQA', 'DQB', 'DPA', 'DPB', 'H-2']
    allele_map = {key: [] for key in allele_keywords}
    for allele in set(alleles):
        for key in allele_keywords:
            if key in allele:
                allele_map[key].append(allele.replace('*', '').replace(':', ''))
                break

    paired_alleles = []
    # DR
    avail_dr_alleles = [allele for allele in avail_alleles if 'DRB' in allele]
    for allele in allele_map['DRB']:
        for j in range(len(avail_dr_alleles)):
            if avail_dr_alleles[j].startswith(allele):
                paired_alleles.append(avail_dr_alleles[j])
                break
    # DQ
    avail_dq_alleles = [allele for allele in avail_alleles if '-DQ' in allele]
    if len(allele_map['DQA']) != 0 and len(allele_map['DQB']) != 0:
        matched = set()
        for dqa in allele_map['DQA']:
            for dqb in allele_map['DQB']:
                for allele in avail_dq_alleles:
                    if dqa in allele and dqb in allele:
                        matched.add(dqa)
                        matched.add(dqb)
                        paired_alleles.append(allele)
                        break
        for allele in matched:
            if allele in allele_map['DQA']:
                allele_map['DQA'].remove(allele)
            else:
                allele_map['DQB'].remove(allele)
    if len(allele_map['DQA']) != 0:
        for dqa in allele_map['DQA']:
            for allele in avail_dq_alleles:
                if dqa in allele:
                    paired_alleles.append(allele)
                    break
    if len(allele_map['DQB']) != 0:
        for dqb in allele_map['DQB']:
            for allele in avail_dq_alleles:
                if dqb in allele:
                    paired_alleles.append(allele)
                    break
    # DP
    avail_dp_alleles = [allele for allele in avail_alleles if '-DP' in allele]
    if len(allele_map['DPA']) != 0 and len(allele_map['DPB']) != 0:
        matched = set()
        for dpa in allele_map['DPA']:
            for dpb in allele_map['DPB']:
                for allele in avail_dp_alleles:
                    if dpa in allele and dpb in allele:
                        matched.add(dpa)
                        matched.add(dpb)
                        paired_alleles.append(allele)
                        break
        for allele in matched:
            if allele in allele_map['DPA']:
                allele_map['DPA'].remove(allele)
            else:
                allele_map['DPB'].remove(allele)
    if len(allele_map['DPA']) != 0:
        for dpa in allele_map['DPA']:
            for allele in avail_dp_alleles:
                if dpa in allele:
                    paired_alleles.append(allele)
                    break
    if len(allele_map['DPB']) != 0:
        for dpb in allele_map['DPB']:
            for allele in avail_dp_alleles:
                if dpb in allele:
                    paired_alleles.append(allele)
                    break
    return paired_alleles

if __name__ == '__main__':
    alleles = prepare_class_II_alleles(['DRB1*04:05', 'DRB1*15:01', 'DRB4*01:03', 'DRB5*01:01', 'DQB1*03:02', 'DQB1*06:02', 'DQA1*01:02', 'DQA1*03:03', 'DPB1*104:01', 'DPB1*04:01', 'DPA1*01:03'])
    print(alleles)
    for allele in alleles:
        print(normalize_allele_name(allele))
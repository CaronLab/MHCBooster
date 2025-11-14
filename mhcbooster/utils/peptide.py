import re
from typing import Union, List, Type
from copy import deepcopy
import numpy as np


common_aa = "ARNDCQEGHILKMFPSTWYV"

def remove_modifications(peptides: Union[List[str], str, Type[np.array]]):
    if isinstance(peptides, str):
        return ''.join(re.findall('[A-Z]+', peptides))
    unmodified_peps = []
    for pep in peptides:
        pep = ''.join(re.findall('[A-Z]+', pep))
        unmodified_peps.append(pep)
    return unmodified_peps

def remove_previous_and_next_aa(peptides: Union[List[str], str, Type[np.array]]):
    peptides = deepcopy(peptides)
    return_one = False
    if isinstance(peptides, str):
        peptides = [peptides]
        return_one = True
    for i in range(len(peptides)):
        if peptides[i][1] == '.':
            peptides[i] = peptides[i][2:]
        if peptides[i][-2] == '.':
            peptides[i] = peptides[i][:-2]
    if return_one:
        return peptides[0]
    return peptides

def get_previous_and_next_aa(peptides: Union[List[str], str, Type[np.array]]):
    prev_aas = np.full(len(peptides), '', dtype=str)
    next_aas = np.full(len(peptides), '', dtype=str)
    if isinstance(peptides, str):
        peptides = [peptides]
    for i in range(len(peptides)):
        if peptides[i][1] == '.' and peptides[i][-2] == '.':
            prev_aas[i] = peptides[i][0]
            next_aas[i] = peptides[i][-1]
    return prev_aas, next_aas

def remove_charge(peptides: Union[List[str], str]):
    peptides = deepcopy(peptides)
    return_one = False
    if isinstance(peptides, str):
        peptides = [peptides]
        return_one = True
    for i in range(len(peptides)):
        if peptides[i][-1].isdigit():
            peptides[i] = peptides[i][:-1]
        if peptides[i][-1] == '/':
            peptides[i] = peptides[i][:-1]
    if return_one:
        return peptides[0]
    return peptides

def get_charge(peptides: Union[List[str], str, Type[np.array]]):
    charges = []
    if isinstance(peptides, str):
        peptides = [peptides]
    for peptide in peptides:
        if peptide[-1].isdigit():
            charges.append(int(peptide[-1]))
    return charges

def replace_uncommon_aas(peptides: Union[List[str], str]):
    peptide_map = {}
    replaced_peptides = deepcopy(peptides)
    if isinstance(replaced_peptides, str):
        for aa in replaced_peptides:
            if aa not in common_aa:
                replaced_peptides = replaced_peptides.replace(aa, 'X')
        peptide_map[peptides] = replaced_peptides
    for i in range(len(replaced_peptides)):
        for aa in replaced_peptides[i]:
            if aa not in common_aa:
                replaced_peptides[i] = replaced_peptides[i].replace(aa, 'X')
        peptide_map[peptides[i]] = replaced_peptides[i]
    return peptide_map


def encode_peptide_modifications(peptides: Union[List[str], Type[np.array]], modification_encoding: dict = None,
                                 return_encoding_dictionary: bool = False):
    peptides = deepcopy(peptides)
    peptides = np.array(peptides)
    peptides = remove_previous_and_next_aa(peptides)

    if modification_encoding is not None:  # a dictionary containing modification encoding was passed
        mod_dict = modification_encoding
    else:  # we need to build the modification encoding dictionary
        modifications = []
        for pep in peptides:
            mods = re.findall('([a-zA-z][[({][0-9.a-zA-Z]+[](}])', pep)
            modifications += mods
        modifications = set(modifications)

        mod_dict = {}
        for i, mod in enumerate(modifications):
            mod_dict[mod] = str(i + 1)

    # replace the modifications in the peptide strings with the number encodings
    for i, pep in enumerate(peptides):
        for mod in mod_dict.keys():
            while mod in pep:
                pep = pep.replace(mod, mod_dict[mod])
        peptides[i] = pep

    if return_encoding_dictionary:
        return peptides, mod_dict
    else:
        return peptides


def clean_peptide_sequences(peptides: List[str]) -> List[str]:
    return remove_modifications(remove_previous_and_next_aa(peptides))


def extract_mods(peptide):
    mods = []
    MOD_PATTERN = re.compile(r'([A-Za-z])\s*\[([+-]?\d+(?:\.\d+)?)\]')
    for m in MOD_PATTERN.finditer(peptide):
        residue = m.group(1)
        mass = m.group(2)
        if residue == 'n':
            mods.append(f'N-term({mass})')
        else:
            chars_before = peptide[:m.start()]
            aa_count_before = sum(1 for ch in chars_before if ch.isupper())
            pos = aa_count_before + 1  # 1-based position
            mods.append(f'{pos}{residue}({mass})')
    return ', '.join(mods)


def extract_mod_mass_diff(peptide):
    mass_diffs = []
    MOD_PATTERN = re.compile(r'([A-Za-z])\s*\[([+-]?\d+(?:\.\d+)?)\]')
    for m in MOD_PATTERN.finditer(peptide):
        mod_mass = float(m.group(2))
        mass_diffs.append(mod_mass)
    return mass_diffs


def convert_mass_diff_to_unimod(peptides: List[str], mass_unimod_map, supported_list=None) -> tuple[List[str], List[dict]]:
    pattern = r'\[([^]]+)]'
    unsolved_mods_list = []
    unimod_peptides = []
    for peptide in peptides:
        positions = [list(m.span()) for m in re.finditer(pattern, peptide)]
        unsolved_mods = {}
        unimod_peptide = peptide
        mass_str_lens = np.array([span[1] - span[0] for span in positions], dtype=int)
        accum_lens = np.insert(np.cumsum(mass_str_lens), 0, 0)
        for i in range(len(positions)-1, -1, -1):
            span = positions[i]
            mod_mass_diff = float(unimod_peptide[span[0] + 1: span[1] - 1])
            round_mass_diff = round(mod_mass_diff, 4)

            if span[1] < len(unimod_peptide) and unimod_peptide[span[1]] == '-':
                span[1] += 1
            if round_mass_diff in mass_unimod_map.keys():
                unimod_number = mass_unimod_map[round_mass_diff]
                mod_aa = unimod_peptide[span[0] - 1] if span[0] > 0 else ''
                mod_key = mod_aa + unimod_number
                if supported_list is None or mod_key in supported_list:
                    if mod_aa == 'n' or mod_aa == '':
                        unimod_peptide = '[UNIMOD:1]-' + unimod_peptide[span[1]:]
                    else:
                        unimod_peptide = unimod_peptide[:span[0] + 1] + 'UNIMOD:' + unimod_number + unimod_peptide[span[1] - 1:]
                else:
                    unimod_peptide = unimod_peptide[:span[0]] + unimod_peptide[span[1]:]
                    unsolved_mods[span[0] - accum_lens[i]] = mod_mass_diff
            else:
                unimod_peptide = unimod_peptide[:span[0]] + unimod_peptide[span[1]:]
                unsolved_mods[span[0] - accum_lens[i]] = mod_mass_diff
        if unimod_peptide[0] == 'n':
            unimod_peptide = unimod_peptide[1:]
        unimod_peptides.append(unimod_peptide)
        unsolved_mods_list.append(unsolved_mods)
    return unimod_peptides, unsolved_mods_list

def get_pos_mod_map(peptide: str):
    pattern = r'\[([^]]+)]'
    positions = [m.span() for m in re.finditer(pattern, peptide)]
    mass_str_lens = np.array([span[1] - span[0] for span in positions], dtype=int)
    if peptide.startswith('n'):
        accum_lens = np.insert(np.cumsum(mass_str_lens) + 1, 0, 0)
    else:
        accum_lens = np.insert(np.cumsum(mass_str_lens), 0, 0)
    mod_map = {}
    for i in range(len(positions) - 1, -1, -1):
        span = positions[i]
        mod_mass_diff = float(peptide[span[0] + 1: span[1] - 1])
        round_mass_diff = round(mod_mass_diff, 4)
        mod_map[span[0] - accum_lens[i]] = round_mass_diff
    return mod_map

def get_pos_unimod_map(peptide: str, mass_unimod_map):
    mod_map = get_pos_mod_map(peptide)
    unimod_map = {}
    undef_mod_map = {}
    for pos, mass_diff in mod_map.items():
        if mass_diff in mass_unimod_map.keys():
            unimod_map[pos] = mass_unimod_map[mass_diff]
        else:
            undef_mod_map[pos] = mass_diff
    return unimod_map, undef_mod_map

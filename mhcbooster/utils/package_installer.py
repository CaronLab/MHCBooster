
import zipfile
import tarfile
import requests
import shutil
import json
import subprocess

from pathlib import Path

from future.moves import sys

target_folder = Path(__file__).parent.parent/'third_party'
target_folder.mkdir(parents=True, exist_ok=True)


def install_msfragger(path):
    print('Installing MSFragger...')
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
        print('MSFragger installed to {}'.format(target_folder))
    if not (target_folder / 'jre-17.0.14').exists():
        with tarfile.open(target_folder / 'jre-17.0.14.tar.gz', 'r:gz') as tar:
            tar.extractall(target_folder)
            print('Java runtime environment installed to {}'.format(target_folder))


def install_ionquant(path):
    print('Installing IonQuant...')
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
        print('IonQuant installed to {}'.format(target_folder))
    if not (target_folder / 'jre-17.0.14').exists():
        with tarfile.open(target_folder / 'jre-17.0.14.tar.gz', 'r:gz') as tar:
            tar.extractall(target_folder)
            print('Java runtime environment installed to {}'.format(target_folder))


def install_autort(path):
    print('Installing AutoRT...')
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    env_exist = False
    output = subprocess.check_output('source ~/.bashrc; conda info --json', text=True, shell=True)
    data = json.loads(output)
    env_paths = data.get("envs", [])
    for env_path in env_paths:
        if Path(env_path).name == 'autort':
            env_exist = True
            print('Conda environment "autort" already exists, skipping installation')
            break
    if not env_exist:
        print('Installing conda environment of AutoRT...')
        subprocess.run('source ~/.bashrc; conda create -n autort python==3.8 -y && conda run -n autort pip install tensorflow==2.6.0 keras==2.6.0 matplotlib pandas scikit-learn numpy psutil protobuf==3.19.6', shell=True)
    print('AutoRT installed to {}'.format(target_folder))


def install_bigmhc(path):
    print('Installing BigMHC...')
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
        print('BigMHC installed to {}'.format(target_folder))


def install_netmhcpan(path):
    print('Installing NetMHCpan...')
    with tarfile.open(path, 'r:gz') as tar:
        tar.extractall(target_folder)

    data_url = 'https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz'
    file_name = data_url.split('/')[-1]
    file_path = target_folder / 'netMHCpan-4.1' / file_name
    print('Downloading data from {}'.format(data_url))
    response = requests.get(data_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully to {file_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    with tarfile.open(file_path, 'r:gz') as tar:
        data_folder = target_folder / 'netMHCpan-4.1' / 'Linux_x86_64'
        if (data_folder / 'data').is_dir():
            shutil.rmtree(data_folder / 'data')
        elif (data_folder / 'data').is_symlink():
            (data_folder / 'data').unlink()
        tar.extractall(data_folder)
        print('NetMHCpan data extracted to {}'.format(data_folder))

    exe_path = target_folder / 'netMHCpan-4.1' / 'netMHCpan'
    with open(exe_path, 'r') as file:
        content = file.read()
    content = content.replace('/net/sund-nas.win.dtu.dk/storage/services/www/packages/netMHCpan/4.1/netMHCpan-4.1', '`dirname $0`')
    with open(exe_path, 'w') as file:
        file.write(content)

    print('NetMHCpan installed to {}'.format(target_folder))


def install_netmhcIIpan(path):
    print('Installing NetMHCIIpan...')
    with tarfile.open(path, 'r:gz') as tar:
        tar.extractall(target_folder)

    exe_path = target_folder / 'netMHCIIpan-4.3' / 'netMHCIIpan'
    with open(exe_path, 'r') as file:
        content = file.read()
    content = content.replace('/tools/src/netMHCIIpan-4.3', '`dirname $0`')
    with open(exe_path, 'w') as file:
        file.write(content)

    print('NetMHCIIpan installed to {}'.format(target_folder))


def install_mixmhc2pred(path):
    print('Installing MixMHC2pred...')
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_folder/'MixMHC2pred-2.0')
        exe_path = str(target_folder/'MixMHC2pred-2.0'/'MixMHC2pred_unix')
        subprocess.run(f'chmod +x {exe_path}', shell=True)
        print('MixMHC2pred installed to {}'.format(target_folder))

def install():
    args = sys.argv
    if len(args) > 1:
        folder_path = Path(args[1])
        if (folder_path / 'MSFragger-4.1.zip').exists():
            install_msfragger(folder_path / 'MSFragger-4.1.zip')
        if (folder_path / 'IonQuant-1.11.11.zip').exists():
            install_ionquant(folder_path / 'IonQuant-1.11.11.zip')
        if (folder_path / 'AutoRT-master.zip').exists():
            install_autort(folder_path / 'AutoRT-master.zip')
        if (folder_path / 'bigmhc-master.zip').exists():
            install_bigmhc(folder_path / 'bigmhc-master.zip')
        if (folder_path / 'netMHCpan-4.1b.Linux.tar.gz').exists():
            install_netmhcpan(folder_path / 'netMHCpan-4.1b.Linux.tar.gz')
        if (folder_path / 'netMHCIIpan-4.3e.Linux.tar.gz').exists():
            install_netmhcIIpan(folder_path / 'netMHCIIpan-4.3e.Linux.tar.gz')
        if (folder_path / 'MixMHC2pred-2.0.zip').exists():
            install_mixmhc2pred(folder_path / 'MixMHC2pred-2.0.zip')

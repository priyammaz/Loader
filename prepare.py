import os
import shutil
import numpy as np
import argparse
import torchvision
import zipfile
import tarfile
import librosa
import soundfile as sf
from PIL import Image, ImageFile
from tqdm import tqdm
import pandas as pd
from utils import download 

ImageFile.LOAD_TRUNCATED_IMAGES = False

#########################
#### VISION DATASETS ####
#########################

def prepare_coco(path_to_root=None):
    link_to_train_data = "http://images.cocodataset.org/zips/train2017.zip"
    link_to_val_data = "http://images.cocodataset.org/zips/val2017.zip"
    link_to_annotations = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    path_to_root = os.path.join(path_to_root, "coco2017")

    if not os.path.isdir(path_to_root):
        os.mkdir(path_to_root)
        for link in [link_to_train_data, link_to_val_data, link_to_annotations]:
            print("Downloading", link)
            path_to_download = os.path.join(path_to_root, link.split("/")[-1])
            download(link, path_to_download, progress_bar=True)

            print("Unpacking Zip File")
            with zipfile.ZipFile(path_to_download, "r") as zip:
                zip.extractall(path_to_root)

            for file in os.listdir(path_to_root):
                if ".zip" in file:
                    os.remove(os.path.join(path_to_root, file))


def prepare_imagenet(path_to_root=None):
    pass

def prepare_tinyimagenet(path_to_root=None):
    pass

def prepare_dogsvscats(path_to_root):
    path_to_download = os.path.join(os.path.dirname(path_to_root), "download.zip")
    if not os.path.isdir(path_to_root):
        link = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
        download(link, path_to_download, progress_bar=True)

        with zipfile.ZipFile(path_to_download, "r") as zip:
            zip.extractall(os.path.dirname(path_to_root))

        ### cleanup zip, txt and pdf files ###
        print("Unpacking ZIP file")
        for file in os.listdir(os.path.dirname(path_to_root)):
            if (".txt" in file) or (".pdf" in file) or (".zip" in file):
                os.remove(os.path.join(os.path.dirname(path_to_root), file))

        ### rename PetImages folder to dogsvscats ###
        os.rename(os.path.join(os.path.dirname(path_to_root), "PetImages"), path_to_root)

        ### Clean Up CatsVDogs ###
        path_to_catvdog = os.path.join(path_to_root)
        path_to_cats = os.path.join(path_to_catvdog, "Cat") # Get Path to Cat folder
        path_to_dogs = os.path.join(path_to_catvdog, "Dog") # Get Path to Dog folder

        dog_files = os.listdir(path_to_dogs) # Get list of all files inside of dog folder
        cat_files = os.listdir(path_to_cats) # Get list of all files inside cat folder

        path_to_dog_files = [os.path.join(path_to_dogs, file) for file in dog_files] # Get full path to each cat file
        path_to_cat_files = [os.path.join(path_to_cats, file) for file in cat_files] # Get full path to each dog file

        path_to_files = path_to_dog_files + path_to_cat_files

        for file in tqdm(path_to_files):
            try:
                img = np.array(Image.open(file))
                if img.shape[-1] != 3:
                    os.remove(file) # Delete image if it doesnt have three channels
            except Exception as e:
                print(e)
                os.remove(file) # Delete if not an image file, or broken image

        

def prepare_mnist(path_to_root):
    if not os.path.isdir(path_to_root):
        trainset = torchvision.datasets.MNIST(path_to_root, train=True, download=True)
        testset = torchvision.datasets.MNIST(path_to_root, train=False, download=True)

def prepare_cifar10(path_to_root):
    if not os.path.isdir(path_to_root):
        trainset = torchvision.datasets.CIFAR10(path_to_root, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(path_to_root, train=False, download=True)

def prepare_cifar100(path_to_root):
    if not os.path.isdir(path_to_root):
        trainset = torchvision.datasets.CIFAR100(path_to_root, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(path_to_root, train=False, download=True)

def prepare_celeba(path_to_root):
    pass

##########################
##### AUDIO DATASETS #####
##########################

def prepare_librispeech(path_to_root):
    pass

def prepare_speechaccentarchive(path_to_root):
    pass

def prepare_coraal(path_to_root):
    def part_number_gen(num_parts):
        """
        Quick helper funtion to iterate link indexes in Coraal dataset
        """
        parts_idx = list(range(1, num_parts+1))
        parts = []
        for i in parts_idx:
            i = str(i)
            if len(i) == 1:
                parts.append(f"0{i}")
            else:
                parts.append(i)
        return parts

    if not os.path.isdir(path_to_root):
        CORAAL_ATL_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part{part}_2020.05.tar.gz" for part in part_number_gen(4)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_textfiles_2020.05.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_metadata_2020.05.txt"}

        CORAAL_DCA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(10)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textfiles_2018.10.06.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_metadata_2018.10.06.txt"}


        CORAAL_DCB_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(14)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_textfiles_2018.10.06.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_metadata_2018.10.06.txt"}

        CORAAL_DTA_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part{part}_2023.06.tar.gz" for part in part_number_gen(10)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_textfiles_2023.06.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_metadata_2023.06.txt"}

        CORAAL_LES_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part{part}_2021.07.tar.gz" for part in part_number_gen(3)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_textfiles_2021.07.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt"}

        CORAAL_PRV_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part{part}_2018.10.06.tar.gz" for part in part_number_gen(4)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_textfiles_2018.10.06.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_metadata_2018.10.06.txt"}

        CORAAL_ROC_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part{part}_2020.05.tar.gz" for part in part_number_gen(5)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_textfiles_2020.05.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_metadata_2020.05.txt"}

        CORAAL_VLD_FILES = {"audio": [f"http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part{part}_2021.07.tar.gz" for part in part_number_gen(4)], 
                            "transcripts": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_textfiles_2021.07.tar.gz", 
                            "metadata": "http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt"}


        coraal = {"atlanta_georgia": CORAAL_ATL_FILES, 
                "washington_dc_1968": CORAAL_DCA_FILES,
                "washington_dc_2016": CORAAL_DCB_FILES, 
                "detroit_michicagn": CORAAL_DTA_FILES, 
                "lower_east_side_new_york": CORAAL_LES_FILES, 
                "princeville_north_carolina": CORAAL_PRV_FILES, 
                "rochester_new_york": CORAAL_ROC_FILES, 
                "valdosta_georgia": CORAAL_VLD_FILES}
        

        location_dirs = ['atlanta_georgia', 'detroit_michicagn', 'lower_east_side_new_york',  \
                              'princeville_north_carolina', 'rochester_new_york', 'valdosta_georgia', \
                              'washington_dc_1968', 'washington_dc_2016']
        
        sr = 16000
        
        ### Create Folder for CORAAL ###
        os.mkdir(path_to_root)

        ### Iterate through all Cities ###
        for location in coraal:
            print(f"Processing {location}")
            path_to_loc_folder = os.path.join(path_to_root, location)
            path_to_audio_store = os.path.join(path_to_loc_folder, "audios")
            path_to_transcript_store = os.path.join(path_to_loc_folder, "transcripts")
            if not os.path.isdir(path_to_loc_folder):
                os.mkdir(path_to_loc_folder)
                os.mkdir(path_to_audio_store)
                os.mkdir(path_to_transcript_store)

            for audio in tqdm(coraal[location]["audio"]):
                audio_file_name = audio.split("/")[-1]
                path_to_audio_file = os.path.join(path_to_audio_store, audio_file_name)

                ### Download Audios ###
                if not os.path.exists(path_to_audio_file):
                    download(audio, path_to_audio_file)
            
                ### Decompress Audios ###
                with tarfile.open(path_to_audio_file) as f:
                    f.extractall(path_to_audio_store)

            ### Download and Decompress Transcript ###
            transcript_link = coraal[location]["transcripts"]
            transcript_name = transcript_link.split("/")[-1]
            path_to_trancript_file = os.path.join(path_to_transcript_store, transcript_name)

            download(transcript_link, path_to_trancript_file)
            with tarfile.open(path_to_trancript_file) as f:
                f.extractall(path_to_transcript_store)

            ### Download MetaData ###
            download(coraal[location]["metadata"], os.path.join(path_to_loc_folder, "metadata.txt"))


            ### Delete and Hidden and Compressed Files ###
            for file in os.listdir(path_to_audio_store):
                if file.startswith(".") or ".tar.gz" in file:
                    os.remove(os.path.join(path_to_audio_store, file))

            for file in os.listdir(path_to_transcript_store):
                if file.startswith(".") or ".tar.gz" in file:
                    os.remove(os.path.join(path_to_transcript_store, file))

        ### Build CSV with All Data and Start/End Times in Audios ###
        datasets = []
        for location in location_dirs:
            audio_root = os.path.join(path_to_root, location, "audios")
            transcription_root = os.path.join(path_to_root, location, "transcripts")
            path_to_metadata = os.path.join(path_to_root, location, "metadata.txt")

            file_roots = [file for file in os.listdir(audio_root) if ".wav" in file]
            file_roots = [file.split(".")[0] for file in file_roots if not file.startswith(".")]
            path_to_files = [(os.path.join(audio_root, f"{file}.wav"), os.path.join(transcription_root, f"{file}.txt")) for file in file_roots]

            cleaned_transcription = []
            for path_to_audio, path_to_transcriptions in path_to_files:
                filetag = path_to_audio.split("/")[-1].split(".")[0]
                transcriptions = pd.read_csv(path_to_transcriptions, sep='\t', lineterminator='\n')
                transcriptions = transcriptions[transcriptions["Spkr"].str.contains("int")==False]
                transcriptions = transcriptions[transcriptions["Content"].str.contains("pause")==False]
                transcriptions = transcriptions[transcriptions["Content"].str.contains("<|>|RD-")==False]
                transcriptions["Content"] = transcriptions["Content"].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()
                transcriptions["total_time"] = transcriptions["EnTime"] - transcriptions["StTime"]
                transcriptions = transcriptions[transcriptions["total_time"] >= 2].reset_index(drop=True)
                transcriptions["start_frame"] = (transcriptions["StTime"]*sr).astype(int)
                transcriptions["end_frame"] = (transcriptions["EnTime"]*sr).astype(int)
                transcriptions["path_to_audio"] = path_to_audio
                transcriptions = transcriptions[["start_frame", "end_frame", "Content", "path_to_audio"]]
                transcriptions["file_id"] = filetag
                cleaned_transcription.append(transcriptions)
            
            transcriptions = pd.concat(cleaned_transcription).reset_index(drop=True) 
            metadata = pd.read_csv(path_to_metadata,sep='\t', lineterminator='\n')
            data = pd.merge(transcriptions, metadata, left_on="file_id", right_on="CORAAL.File", how="left")
            selected_columns = ["start_frame", "end_frame", "Content", "path_to_audio", 
                                "file_id", "Gender", "Age", "Occupation"]
            
            data = data[selected_columns]
            data.columns = ["start_frame", "end_frame", "transcription", "path_to_audio", 
                            "file_id", "gender", "age", "occupation"]
            
            data["location"] = " ".join(location.split("_"))
            data["accent"] = "african_american"

            datasets.append(data)

        datasets = pd.concat(datasets).reset_index(drop=True)

        ### Create Unique Identifier for Each Audio File ###
        datasets["audio_id"] = ""
        for grouper, group in datasets.groupby("file_id"):
            datasets.loc[datasets["file_id"] == grouper, "audio_id"] = group["file_id"] + [f"_{i+1}" for i in range(len(group))]
            

        ### Iterate Through CSV and Slice up Audios ###
        print("Cutting Up Audios!!")

        datasets["path_to_audio_split"] = ""

        for grouper, group in tqdm(datasets.groupby("path_to_audio")):
            path_to_split_audio = os.path.join(path_to_root, "split_audio")
            audio = librosa.load(grouper, sr=sr)[0]

            
            if not os.path.isdir(path_to_split_audio):
                os.mkdir(path_to_split_audio)

            for idx, row in group.iterrows():
                audio_start, audio_end = row.start_frame, row.end_frame
                path_to_audio_store = os.path.join(path_to_split_audio, f"{row.audio_id}.wav")
                datasets.loc[idx, "path_to_audio_split"] = path_to_audio_store

                audio_split = audio[audio_start:audio_end]

                sf.write(path_to_audio_store, audio_split, sr)

        ### Delete Longform Audios ###
        for folder in os.listdir(path_to_root):
            if os.path.isdir(os.path.join(path_to_root, folder)) and folder != "split_audio":
                shutil.rmtree(os.path.join(path_to_root, folder),ignore_errors=False, onerror=None)
            
                
        ### Save Datasets as Metadata ###
        datasets.to_csv(os.path.join(path_to_root, "metadata.csv"), index=False)


def prepare_l2arctic(path_to_root):
    pass

def prepare_edacc(path_to_root):
    pass


#########################
### LANGUAGE DATASETS ###
#########################

def prepare_openwebtext(path_to_root=None):
    pass

def prepare_wikipedia(path_to_root=None):
    pass

#########################
## MULTIMODAL DATASETS ##
#########################

def prepare_coco(path_to_root=None):
    link_to_train_data = "http://images.cocodataset.org/zips/train2017.zip"
    link_to_val_data = "http://images.cocodataset.org/zips/val2017.zip"
    link_to_annotations = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    path_to_root = os.path.join(path_to_root, "coco2017")

    if not os.path.isdir(path_to_root):
        os.mkdir(path_to_root)
        for link in [link_to_train_data, link_to_val_data, link_to_annotations]:
            print("Downloading", link)
            path_to_download = os.path.join(path_to_root, link.split("/")[-1])
            download(link, path_to_download, progress_bar=True)

            print("Unpacking Zip File")
            with zipfile.ZipFile(path_to_download, "r") as zip:
                zip.extractall(path_to_root)

            for file in os.listdir(path_to_root):
                if ".zip" in file:
                    os.remove(os.path.join(path_to_root, file))
                    
def prepare_flikr8k(path_to_root):
    pass

def prepare_flikr30k(path_to_root):
    pass


########################
#### DEFINE CATALOG ####
########################

dataset_catalog = {
  
    "vision_datasets": {

        "imagenet": {"prepare": prepare_imagenet, "src": "huggingface"}, 
        "tinyimagenet": {"prepare": prepare_tinyimagenet, "src": "huggingface"}, 
        "dogsvscats": {"prepare": prepare_dogsvscats, "src": "wget"}, 
        "mnist": {"prepare": prepare_mnist, "src": "torchvision"},
        "cifar10": {"prepare": prepare_cifar10, "src": "torchvision"},
        "cifar100": {"prepare": prepare_cifar100, "src": "torchvision"},
        "celeba": {"prepare": prepare_celeba, "src": "kaggle"}

    }, 
    
    "audio_datasets": {
        
        "librispeech": {"prepare": prepare_librispeech, "src": "wget"}, 
        "speechaccentarchive": {"prepare": prepare_speechaccentarchive, "src": "kaggle"},
        "coraal": {"prepare": prepare_coraal, "src": "wget"}, 
        "l2arctic": {"prepare": prepare_l2arctic, "src": "wget"}, 
        "edacc": {"prepare": prepare_edacc, "src": "wget"}

    }, 

    "language_datasets": {

        "openwebtext": {"prepare": prepare_openwebtext, "src": "huggingface"}, 

    }, 
    
    "multimodal_datasets": {

        "flikr8k": {"prepare": prepare_flikr8k, "src": "kaggle"}, 
        "coco2017": {"prepare": prepare_coco, "src": "wget"}

    }

}

def get_dataset_groups(dataset_catalog):
    return list(dataset_catalog.keys())

def get_dataset_names(dataset_catalog):
    names = []
    for key in get_dataset_groups(dataset_catalog):
        names.extend(dataset_catalog[key].keys())
    return names

def prepare(dataset_catalog,
            path_to_root, 
            include_groups=None, 
            exclude_groups=None, 
            include_datasets=None, 
            exclude_datasets=None,
            hf_cache_at_root=True):
    
    if include_groups is not None and exclude_groups is not None:
        raise AssertionError("You can either include data groups or exclude data groups, NOT BOTH!!")
    
    if include_datasets is not None and exclude_datasets is not None:
        raise AssertionError("You can either include datasets or exclude datasets, NOT BOTH!!")

    if include_groups is not None:
        for key in dataset_catalog.copy(): 
            if key not in include_groups:
                dataset_catalog.pop(key)
    
    if exclude_groups is not None:
        for key in dataset_catalog.copy(): 
            if key in exclude_groups:
                dataset_catalog.pop(key)

    if include_datasets is not None:
        for key in dataset_catalog.copy():
            for data_key in dataset_catalog[key].copy():
                if data_key not in include_datasets:
                    dataset_catalog[key].pop(data_key)
    
    if exclude_datasets is not None:
        for key in dataset_catalog.copy():
            for data_key in dataset_catalog[key].copy():
                if data_key in exclude_datasets:
                    dataset_catalog[key].pop(data_key)

    
    for group in get_dataset_groups(dataset_catalog):
        for dataset in dataset_catalog[group]:
            dataset_source = dataset_catalog[group][dataset]["src"]
            prepare_func = dataset_catalog[group][dataset]["prepare"]

            ### If a Huggingface dataset, we will store in a Huggingface Cache Folder (Our own or default) ###
            if dataset_source == "huggingface":
                path = os.path.join(path_to_root, "huggingface_cache")
                prepare_func(path_to_root = None if hf_cache_at_root else path)
                
            else:
                path = os.path.join(path_to_root, dataset)
                prepare_func(path_to_root = path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Preparation")

    parser.add_argument("--root", type=str, required=True,
                        help='Path to root directory for all datasets')
    
    parser.add_argument('--include_groups', nargs='+', choices=(get_dataset_groups(dataset_catalog)))

    parser.add_argument('--exclude_groups', nargs="+", choices=(get_dataset_groups(dataset_catalog)))

    parser.add_argument('--include_datasets', nargs="+", choices=(get_dataset_names(dataset_catalog)))

    parser.add_argument('--exclude_datasets', nargs="+", choices=(get_dataset_names(dataset_catalog)))

    args = parser.parse_args()

    prepare(dataset_catalog=dataset_catalog,
            path_to_root=args.root, 
            include_groups=args.include_groups, 
            exclude_groups=args.exclude_groups, 
            include_datasets=args.include_datasets, 
            exclude_datasets=args.exclude_datasets)

    

    


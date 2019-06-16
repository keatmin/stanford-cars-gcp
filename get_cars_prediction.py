from fastai.vision import *
from fastai import * 
import requests
import shutil
import fire

model_url = 'https://drive.google.com/uc?export=download&confirm=ZfKU&id=1SiT2sHE6JDokx7m3qdlMTJJ0uAv5F73i'
export_file_name = '24class.pkl'
path = Path(__file__).parent

def download_file(url,dest): 
    if dest.exists():
        return print('Pickle file already exists,')
    print('Downloading model')
    data = requests.get(url,stream=True)
    print("Saving model")
    with open(dest, 'wb') as f: 
        shutil.copyfileobj(data.raw, f)
        print('Model saved')

def setup_learner(data):
    #download_file(model_url, path/export_file_name)
    learn = load_learner(path, export_file_name, test=data)
    return learn

def setup_images(test_path): 
    """
    Could use from_df or from_csv if these are used
    """
    src = ImageList.from_folder(test_path)
    return src

def generate_csv(learn, preds,test_path,fname): 
    lst = [(learn.data.classes[np.argmax(p).item()],max(p).item())for p in preds[0]]
    pred_df = pd.DataFrame(lst, columns=['class', 'prob'])
    g = pd.Series(list(Path(test_path).iterdir()),name='filename').map(lambda x: str(x).split('/')[-1])
    joint_df = pred_df.join(g)
    return joint_df.to_csv(fname,index=False)

def analyze(test_path,csv_fname='preds.csv'): 
    defaults.device = torch.device('cpu')
    prediction_data = setup_images(test_path)
    print('Setting up learner...')
    learn = setup_learner(prediction_data)
    print('Predicting images...')
    preds = learn.get_preds(DatasetType.Test)
    print('Generating csv...')
    generate_csv(learn, preds, test_path, csv_fname)
    return print(f'{csv_fname} generated..')

if __name__ == '__main__':    fire.Fire(analyze)
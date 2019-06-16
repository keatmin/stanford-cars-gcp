from fastai.vision import *
from fastai import * 
import fire

model_url = 'https://drive.google.com/uc?export=download&confirm=63dP&id=1ZY9yt5Gtkvoy4HEtEqFVjZzopGLaMOPq'
export_file_name = '152resnet.pkl'
path = Path(__file__).parent

def setup_learner(data ,model):
    learn = load_learner(path, model, test=data)
    return learn

def setup_images(test_path): 
    src = ImageList.from_folder(test_path)
    return src

def generate_csv(learn, preds,test_path,fname): 
    lst = [(learn.data.classes[np.argmax(p).item()],max(p).item())for p in preds[0]]
    pred_df = pd.DataFrame(lst, columns=['prediction', 'probability'])
    g = pd.Series(list(Path(test_path).iterdir()),name='filename').map(lambda x: str(x).split('\\')[-1])
    joint_df = pred_df.join(g)[['filename','prediction','probability']]
    return joint_df.to_csv(fname,index=False)

def analyze(test_path,csv_fname='preds.csv',model=export_file_name): 
    defaults.device = torch.device('cpu')
    prediction_data = setup_images(test_path)
    print('Setting up learner...')
    learn = setup_learner(prediction_data,model)
    print('Predicting images...')
    preds = learn.get_preds(DatasetType.Test)
    print('Generating csv...')
    generate_csv(learn, preds, test_path, csv_fname)
    return print(f'{csv_fname} generated..')

if __name__ == '__main__':    fire.Fire(analyze)

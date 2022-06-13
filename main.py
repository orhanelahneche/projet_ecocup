from training import *
from data_preparation import *
from prediction import * 
from tqdm import tqdm

#prepare_training_data()
#clf = train_model()
#clf = load_model(Path.cwd() / "model(50, 50).sav")
#detect_false_negs(clf, Path.cwd() / "train" / "images" / "neg")
#retrain_model()
clf = load_model(Path.cwd() / "model_retrained(50, 50).sav")
for img in (Path.cwd() / "test").glob("*.jpg"):
    fenetre_glissante(io.imread(img), clf)
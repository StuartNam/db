from metrics import FDFR, ISM

DATA_FOLDER = './data/valid/'

print(FDFR.eval(DATA_FOLDER))
print(ISM.eval(DATA_FOLDER, DATA_FOLDER))




import numpy as np
import io
from keras.models import load_model, Model, Input
from keras.layers import Average, Lambda

def ensemble(model_file_list):

    models = [ load_model('drive/app/en/'+model_file) for model_file in model_file_list ]

    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    model = ensembleModels(models, model_input)

    model.save('drive/app/en/model_ensemble.h5')

def ensembleModels(models, model_input):
    y_models = [model(model_input) for model in models] 

    y_avg = Average()(y_models) 
    model_ensemble = Model(inputs=model_input, outputs=y_avg, name='ensemble')  
   
    return model_ensemble

def main():
    model_list = ['model700.h5', 'model693.h5', 'model690.h5']
    ensemble(model_list)
    print('finished')

if __name__=="__main__":
    main()

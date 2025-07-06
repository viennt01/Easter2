import config
import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
from data_loader import data_loader
import tensorflow.keras.backend as K

def ctc_custom(args):
    y_pred, labels, input_length, label_length = args
    
    ctc_loss = K.ctc_batch_cost(
        labels, 
        y_pred, 
        input_length, 
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha=0.25 
    return alpha*(K.pow((1-p),gamma))*ctc_loss

# def load_easter_model(checkpoint_path):
#     if checkpoint_path == "Empty":
#         checkpoint_path = config.BEST_MODEL_PATH
#     try:
#         checkpoint = tensorflow.keras.models.load_model(
#             checkpoint_path,
#             custom_objects={'<lambda>': lambda x, y: y,
#             'tensorflow':tf, 'K':K}
#         )
        
#         EASTER = tensorflow.keras.models.Model(
#             checkpoint.get_layer('the_input').input,
#             checkpoint.get_layer('Final').output
#         )
#     except:
#         print ("Unable to Load Checkpoint.")
#         return None
#     return EASTER
def load_easter_model(checkpoint_path):
    if checkpoint_path == "Empty":
        checkpoint_path = config.BEST_MODEL_PATH
        print("Using Default Checkpoint Path:", checkpoint_path)
    try:
        print("Loading Checkpoint from:", checkpoint_path)
        checkpoint = tensorflow.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                'ctc_custom': ctc_custom,
                '<lambda>': lambda x, y: y,
                'tensorflow': tf,
                'K': K
            }
        )
        
        EASTER = tensorflow.keras.models.Model(
            checkpoint.get_layer('the_input').input,
            checkpoint.get_layer('Final').output
        )
    except Exception as e:
        print("Unable to Load Checkpoint.")
        print("Reason:", e)
        return None
    return EASTER
    
def decoder(output,letters):
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j,:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret
    
def test_on_iam(show = True, partition='test', uncased=False, checkpoint="Empty"):
    
    print ("loading metdata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()
    charlist = training_data.charList
    print ("loading checkpoint...")
    print ("calculating results...")
    
    model = load_easter_model(checkpoint)
    char_error = 0
    total_chars = 0
    
    batches = 1
    output_file = open("test_results.txt", "w", encoding='utf-8')

    # Nếu muốn ghi ra file CSV, bật đoạn này:
    import csv
    csv_file = open("results.csv", "w", newline="", encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Ground Truth", "Prediction", "Edit Distance"])
    while batches > 0:
        batches = batches - 1
        if partition == 'validation':
            print ("Using Validation Partition")
            imgs, truths, _ = validation_data.getValidationImage()
        else:
            print ("Using Test Partition")
            imgs,truths,_ = test_data.getTestImage()

        print ("Number of Samples : ",len(imgs))
        for i in tqdm(range(0,len(imgs))):
            img = imgs[i]
            truth = truths[i].strip(" ").replace("  "," ")
            output = model.predict(img, verbose=0)
            prediction = decoder(output, charlist)
            output = (prediction[0].strip(" ").replace("  ", " "))
            ed = edit_distance(output.lower(), truth.lower()) if uncased else edit_distance(output, truth)
            char_error += ed
            # if uncased:
            #     char_error += edit_distance(output.lower(),truth.lower())
            # else:
            #     char_error += edit_distance(output,truth)
                
            total_chars += len(truth)
            
            
            # Ghi ra file
            # output_file.write(f"Ground Truth : {truth}\n")
            # output_file.write(f"Prediction [{ed}]  : {output}\n")
            # output_file.write("*" * 50 + "\n")

            # Ghi ra CSV nếu bật
            # csv_writer.writerow([truth, output, ed])


            # if show:
            #     print ("Ground Truth :", truth)
            #     print("Prediction [",edit_distance(output,truth),"]  : ",output)
            #     print ("*"*50)
    print ("Character error rate is : ",(char_error/total_chars)*100)
    output_file.write(f"\nCharacter error rate is : {(char_error/total_chars)*100}%\n")
    output_file.close()
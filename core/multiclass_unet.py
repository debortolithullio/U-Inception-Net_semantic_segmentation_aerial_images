from .unet import *
from .utils import *
from keras.models import load_model
from sklearn.metrics import recall_score, precision_score

class MulticlassUNET():
    def __init__(self, classes, input_size = 224):
        '''
            :param classes: List with all classes to predict
        '''
        self.classes = classes
        self.INPUT_SIZE = input_size

        self.models = {}
        for cl in classes:
            self.models[cl] = UNET_binary_model()

    def load_UNET_model(self, class_name, model_file_path,):
        '''
            Function to load pre-trained Keras Model
            :param class_name: Name of the class to load the model
            :param model_file_path: path of the h5 file containing the model architecture and weights
        '''
        self.models[class_name].load_weights(model_file_path)

    def predict_greater_image(self, class_name, image):
        '''
            Function to predict an image greater than the INPUT_SIZE
            :param class_name: Name of the class to train
            :param image: image to be predicted
        '''
        img_shape = image.shape
        chunks = divide_into_chunks(image, self.INPUT_SIZE)
        for i in tqdm(range(len(chunks))):
            chunks[i] = self.models[class_name].predict(preprocess_input(chunks[i].reshape(1, self.INPUT_SIZE, self.INPUT_SIZE, 3)))

        size_image = int(np.sqrt(len(chunks))*self.INPUT_SIZE)
        mask_pred = np.zeros((size_image, size_image))

        chunk_idx = 0
        for i in range(0, mask_pred.shape[0], self.INPUT_SIZE):
            for j in range(0, mask_pred.shape[1], self.INPUT_SIZE):
                mask_pred[i:i+self.INPUT_SIZE, j:j+self.INPUT_SIZE] = chunks[chunk_idx].reshape(self.INPUT_SIZE, self.INPUT_SIZE)
                chunk_idx += 1

        return cv2.resize(np.round(mask_pred), (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA).astype(int)

    def train_UNET_model_generator(self, class_name, optimizer, training_batch_generator, num_epochs, val_batch_generator, queue_size, callbacks):
        '''
            Function to train Keras Model
            :param class_name: Name of the class to train
            :param training_batch_generator: training batch generator
            :param num_epochs: number of epochs to train
            :param queue_size: size of queue to use in training
            :param callbacks: list of callbacks to use in training
        '''

        self.models[class_name].compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

        hist = self.models[class_name].fit_generator(generator=training_batch_generator,
                           epochs=num_epochs,
                           verbose=1,
                           validation_data=val_batch_generator,
                           max_queue_size=queue_size,
                           workers=-1,
                           callbacks=callbacks)

        return hist

    def test_UNET_model(self, class_name, X_test, y_test, threshold = 0.1):
        '''
            Function to test the model and get a result (Dice Coefficient)
            :param class_name: Name of the class to calculate the metric
            :param X_test: List with paths with test images
            :param y_test: List with paths with test masks
            :return: avg_dice_coef, avg_recall, avg_precision and percentual_of_class(percentual of pixels that is 1 (class))
        '''
        dice_coef_tot = 0
        recall_tot = 0
        precision_tot = 0
        percentual_of_class_tot = 0
        y_true_with_true_labels = 0
        pred_with_true_labels = 0
        function_to_classify = lambda x: 1 if x > threshold else 0
        function_to_classify = np.vectorize(function_to_classify)

        for img_path, mask_path in tqdm(zip(X_test, y_test)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)
            
            mask = np.round(preprocess_input(mask)).ravel()

            pred = self.models[class_name].predict(preprocess_input(img.reshape(1, self.INPUT_SIZE, self.INPUT_SIZE, 3)))
            pred = function_to_classify(pred[0]).astype(np.uint8).ravel()

            dice_coef_tot += dice_coef_float(mask, pred)
            percentual_of_class_tot += mask.sum()/mask.shape[0]
            if mask.sum() > 0:
                recall_tot += recall_score(mask, pred)
                y_true_with_true_labels += 1

            if pred.sum() > 0:
                precision_tot += precision_score(mask, pred)
                pred_with_true_labels += 1
            
        
        return dice_coef_tot/X_test.shape[0], recall_tot/y_true_with_true_labels, precision_tot/pred_with_true_labels, percentual_of_class_tot/X_test.shape[0]
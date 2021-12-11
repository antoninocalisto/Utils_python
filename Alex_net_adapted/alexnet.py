import tensorflow
from keras.models import Sequential, load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model
import numpy, glob, os, logging, time, cv2
import SimpleITK as sitk

numpy.random.seed(1337)

gpu_memory_fraction = 0.7 # Fraction of GPU memory to use
config = tensorflow.ConfigProto(gpu_options=
                       tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))                          
sess = tensorflow.Session(config=config)

# Carregando imagens
print("Carregando imagens")

# Diretorios de imagens
pathFetusTrainSet = "C:/Users/usuario/Pictures/Fetus/"
pathNonFetusTrainSet = "C:/Users/usuario/Pictures/Non_fetus/"

pathFetusValidationSet = "C:/Users/usuario/Pictures/Fetus/"
pathNonValidationTestSet = "C:/Users/usuario/Pictures/Non_fetus/"

pathFetusTestSet = "C:/Users/usuario/Pictures/Fetus/"
pathNonFetusTestSet = "C:/Users/usuario/Pictures/Non_fetus/"

# Pegando nome das imagens da classe positiva e negativa
trainFetusList = glob.glob( pathFetusTrainSet + "*.nii")
trainNonFetusList = glob.glob( pathNonFetusTrainSet + "*.nii")

validationFetusList = glob.glob( pathFetusValidationSet + "*.nii")
validationNonFetusList = glob.glob( pathNonValidationTestSet + "*.nii")

testFetusList = glob.glob( pathFetusTestSet + "*.nii")
testNonFetusList = glob.glob( pathNonFetusTestSet + "*.nii")

# Lendo todas as imagens e salvando conteudo
trainImageDataList = []
validationImageDataList = []
testImageDataList = []

trainClassesList = []
validationClassesList = []
testClassesList = []

# Imagens de treino    
for filename in trainNonFetusList:
    #image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    trainImageDataList.append( image )
    trainClassesList.append(0)

for filename in trainFetusList:
    # image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    trainImageDataList.append( image )
    trainClassesList.append(1)

# Imagens de validacao
for filename in validationNonFetusList:
    # image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    validationImageDataList.append( image ) 
    validationClassesList.append(0)

for filename in validationFetusList:
    # image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    validationImageDataList.append( image )
    validationClassesList.append(1)

# Imagens de teste
for filename in testNonFetusList:
    # image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    testImageDataList.append( image )
    testClassesList.append(0)

for filename in testFetusList:
    # image = cv2.imread(filename, 0)
    image = sitk.ReadImage(filename)
    testImageDataList.append( image )
    testClassesList.append(1)

# Criando e formatando dataset
print("\nCriando e formatando dataset")

# Convertendo pra array numpy e formatando dados
numberOfClasses = 2
Y_train = np_utils.to_categorical(trainClassesList, numberOfClasses)
Y_validation = np_utils.to_categorical(validationClassesList, numberOfClasses)
Y_test = np_utils.to_categorical(testClassesList, numberOfClasses)

imageRows = 159
imageCollumns = 288
imageChannels = 4

trainSamplesList = numpy.array( trainImageDataList, dtype=numpy.uint8 ) 
trainSamplesList = trainSamplesList.reshape( trainSamplesList.shape[0], imageRows, imageCollumns, imageChannels)
trainSamplesList = trainSamplesList.astype( 'float32' )
trainSamplesList /= 255
print(trainSamplesList.shape)

validationSamplesList = numpy.array( validationImageDataList, dtype=numpy.uint8 ) 
validationSamplesList = validationSamplesList.reshape( validationSamplesList.shape[0], imageRows, imageCollumns, imageChannels)
validationSamplesList = validationSamplesList.astype( 'float32' )
validationSamplesList /= 255
print(validationSamplesList.shape)

testSamplesList = numpy.array( testImageDataList, dtype=numpy.uint8 ) 
testSamplesList = testSamplesList.reshape( testSamplesList.shape[0], imageRows, imageCollumns, imageChannels)
testSamplesList = testSamplesList.astype( 'float32' )
testSamplesList /= 255 
print(testSamplesList.shape)

# Configurando CNN
print("\nCarregando a configuracao do treinamento")

numberOfEpochs = 1
batchSize = 128

nbFilter1 = 96
nbFilter2 = 256
nbFilter3 = 394
nbFilter4 = 394
nbFilter5 = 256

numberOfClasses = 2

input1 = Input(shape=(imageRows, imageCollumns,imageChannels))
model1_conv1 = Conv2D(nbFilter1, (3, 3), padding='same')(input1)
model1_conv1 = Activation('relu')(model1_conv1)
model1_pool1 = MaxPooling2D(pool_size=(2, 2))(model1_conv1) 
model1_conv2 = Conv2D(nbFilter2, (3, 3), padding='same')(model1_pool1)
model1_conv2 = Activation('relu')(model1_conv2)
model1_pool2 = MaxPooling2D(pool_size=(2, 2))(model1_conv2)
model1_conv3 = Conv2D(nbFilter3, (3, 3), padding='same')(model1_pool2)
model1_conv3 = Activation('relu')(model1_conv3)
model1_conv4 = Conv2D(nbFilter4, (3, 3), padding='same')(model1_conv3)
model1_conv4 = Activation('relu')(model1_conv4)
model1_conv5 = Conv2D(nbFilter5, (3, 3), padding='same')(model1_conv4)
model1_conv5 = Activation('relu')(model1_conv5)
model1_pool3 = MaxPooling2D(pool_size=(2, 2))(model1_conv5) 

mlp_layers = model1_pool3
mlp_layers = Flatten()(mlp_layers)
mlp_layers = Dense(4096)(mlp_layers) 
mlp_layers = Activation('relu')(mlp_layers)
mlp_layers = Dropout(0.5)(mlp_layers) 
mlp_layers = Dense(4096)(mlp_layers) 
mlp_layers = Activation('relu')(mlp_layers)
mlp_layers = Dropout(0.5)(mlp_layers) 
mlp_layers = Dense(numberOfClasses)(mlp_layers)
mlp_layers = Activation('softmax')(mlp_layers) 


mlp = Model(inputs=[input1], outputs=[mlp_layers])

# Treinando
print("\nTreinando")

file_epochs = open('Epochs.txt', 'w')

mlp.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

filepath = 'model.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

train = mlp.fit(trainSamplesList, Y_train, 
	batch_size = batchSize, 
	epochs = numberOfEpochs, 
	verbose = 1, 
	validation_data=(validationSamplesList, Y_validation))
	#callbacks = [checkpoint]) 

score = mlp.evaluate(validationSamplesList, Y_validation, verbose = 0)

file_epochs.write(str(train.history) + '\n\n')

file_epochs.write('Validation Loss and Validation Accuracy: ')

file_epochs.write(str(score) + '\n\n')

file_epochs.close()

bestModel = load_model('model.h5')

result = open('Result.txt', 'w')

# Testando e calculando estatisticas de desempenho para a base de validacao
result.write("Testando e calculando estatisticas de desempenho para a base de validacao")

predictionsValidationList = []
predictionsValidationList = bestModel.predict(validationSamplesList)
predictionsValidationList = numpy.argmax(predictionsValidationList, axis=1)

predictionsValidationFile = open('ClassificacaoValidacao.txt', 'w')

vp, fp, vn, fn = 0, 0, 0, 0      

for i in range(validationSamplesList.shape[0]):
	predictionsValidationFile.write(str(predictionsValidationList[i]) + "\n")
	if validationClassesList[i] == 0 and predictionsValidationList[i] == 0:
		vn += 1
	elif validationClassesList[i] == 0 and predictionsValidationList[i] == 1:
		fp += 1
	elif validationClassesList[i] == 1 and predictionsValidationList[i] == 1:
		vp += 1
	else:
		fn += 1

sensibility = float( ( vp * 1.0 ) / ( vp + fn ) ) 
specificity = float( ( vn * 1.0 ) / ( vn + fp ) )
accuracy = float( ( (vp + vn) * 1.0 ) / ( vn + fp + vp + fn) )  

result.write("\n\nSensibilidade no conjunto de validacao: " + str(sensibility))
result.write("\nEspecificidade no conjunto de validacao: " + str(specificity))
result.write("\nAcuracia no conjunto de validacao: " + str(accuracy))

predictionsValidationFile.close()

# Testando e calculando estatisticas de desempenho para a base de teste
result.write("\nTestando e calculando estatisticas de desempenho para a base de teste")

predictionsTestList = []
predictionsTestList = bestModel.predict(testSamplesList)
predictionsTestList = numpy.argmax(predictionsTestList,axis=1)

predictionsTestFile = open('ClassificacaoTeste.txt', 'w')

vp, fp, vn, fn = 0, 0, 0, 0

for i in range(testSamplesList.shape[0]):
	predictionsTestFile.write(str(predictionsTestList[i]) + "\n")
	if testClassesList[i] == 0 and predictionsTestList[i] == 0:
		vn += 1
	elif testClassesList[i] == 0 and predictionsTestList[i] == 1:
		fp += 1
	elif testClassesList[i] == 1 and predictionsTestList[i] == 1:
		vp += 1
	else:
		fn += 1

sensibilityTest = float( ( vp * 1.0 ) / ( vp + fn ) ) 
specificityTest = float( ( vn * 1.0 ) / ( vn + fp ) )
accuracyTest = float( ( (vp + vn) * 1.0 ) / ( vn + fp + vp + fn) )  

result.write("\n\nSensibilidade no conjunto de teste: " + str(sensibilityTest))
result.write("\nEspecificidade no conjunto de teste: " + str(specificityTest))
result.write("\nAcuracia no conjunto de teste: " + str(accuracyTest))

predictionsTestFile.close()

result.close()




#Build the model (Attention-56)

#Model Input
ModelInput = layers.Input(shape=(32, 32, 3))

# Conv1
X = layers.Conv2D(32, (5, 5), padding='same')(ModelInput)
X = layers.BatchNormalization()(X)
X = layers.Activation('relu')(X)

#Max pooling
X = layers.MaxPool2D(pool_size=(2, 2))(X)

#Residual Unit 1
X = residual_unit(X, filter_num_first=32, filter_num_final=128, stride=1)

#Attention Module 1
X = attention_module(X, filter_num_first=128, filter_num_final=128, depth=2)

#Residual Unit 2
X = residual_unit(X, filter_num_first=128, filter_num_final=256, stride=2)

#Attention Module 2
X = attention_module(X, filter_num_first=256, filter_num_final=256, depth=1)
X = attention_module(X, filter_num_first=256, filter_num_final=256, depth=1)

#Residual Unit 3
X = residual_unit(X, filter_num_first=256, filter_num_final=512, stride=2)

#Attention Module 3
X = attention_module(X, filter_num_first=512, filter_num_final=512)
X = attention_module(X, filter_num_first=512, filter_num_final=512)
X = attention_module(X, filter_num_first=512, filter_num_final=512)

#Residual Unit 4
X = residual_unit(X, filter_num_first=512, filter_num_final=1024)

X = residual_unit(X, filter_num_first=1024, filter_num_final=1024, short_cut=False)
X = residual_unit(X, filter_num_first=1024, filter_num_final=1024, short_cut=False)

X = layers.BatchNormalization()(X)
X = layers.Activation('relu')(X)
X = layers.AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(X)

#FC
#X = layers.Dropout(0.5)(X)
X = layers.Flatten()(X)
ModelOutput = layers.Dense(10, activation='softmax')(X)

model_attention_92_cifar10 = models.Model(ModelInput, ModelOutput)
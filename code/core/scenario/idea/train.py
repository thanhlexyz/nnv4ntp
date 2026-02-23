#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Train 5x5 to predict center cell
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Set the gpu card number to use. Use this line if your machine has multiple gpu cards.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for FATAL logs only


# In[ ]:


# Parameters
nr = 5 # grid size of nr*nr
Capacity_forecasting = 1 # 1: Capacity forecasting, 0: traffic forecasting
lookback = 3 # history length
alpha = 0.1 # calibration parameter for capacity forecasting

if Capacity_forecasting == 1:
    damage = 'capacity_forecasting'
else:
    damage = 'mae'

cell = 5060 # center cell
cells = get_rows(cell, 21) # returns cell ids in a 21*21 grid


# In[ ]:


# DeepCog architecture from: https://github.com/wnlUc3m/deepcog
sample_shape = (nr, nr)
batch_size = 128
num_cluster = 1
no_epochs = 150
validation_split = 0.2
verbosity = 1
#steps_per_epoch = 40
neurons= 32
ker_sz= 3

def make_nn_model(load_input_shape, lookback, num_cluster):
    '''Build DeepCog architecture'''
    inputs = tf.keras.layers.Input(shape=(
        lookback,  load_input_shape[0],load_input_shape[1],  1))
    x = tf.keras.layers.Conv3D(neurons, kernel_size=(ker_sz, ker_sz, ker_sz), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = tf.keras.layers.Conv3D(neurons, kernel_size=(ker_sz * 2, ker_sz * 2, ker_sz * 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv3D(neurons / 2, kernel_size=(ker_sz * 2,ker_sz * 2, ker_sz * 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(neurons * 2, activation='relu')(x)
    x = tf.keras.layers.Dense(neurons, activation='relu')(x)
    output = tf.keras.layers.Dense(num_cluster)(x)
    model = Model(inputs=inputs, outputs=output)
    if Capacity_forecasting == False:
        model.compile(optimizer=Adam(0.0005), loss = mae)
    else:
        model.compile(optimizer=Adam(0.0005), loss = cost_func_more_args(alpha))
    return model


# In[ ]:


# model summary
model = make_nn_model(sample_shape, lookback, num_cluster)
model.summary()


# In[ ]:


for cell in cells:
    print('CELL',cell)
    roww = []
    dtt=[]
    dt = []
    t = []
    dt_scaled_old=[]
    train_dt = []
    test_dt = []
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    dt_reshaped = []
    roww = get_rows(cell, nr)       
    for i in range(0,len(roww)): 
        rows=roww[i]
        dt.append(pd.read_csv('../Datasets/Milan/PerBS/%s.txt.csv' % rows, header=0, sep=',')) 
        dt[i].columns = ['ID','time','CC','smsin','smsout','callin','callout','internet']
        dt[i] = dt[i].replace(r'^\s*$', np.nan, regex=True)
        val=dt[i]['ID'].iloc[0]
        dt[i].drop('smsin', inplace=True, axis=1)
        dt[i].drop('smsout', inplace=True, axis=1)
        dt[i].drop('callin', inplace=True, axis=1)
        dt[i].drop('callout', inplace=True, axis=1)
        dt[i].drop('ID', inplace=True, axis=1)
        dt[i].drop('CC', inplace=True, axis=1)
        dt[i]['time'] = pd.to_datetime(dt[i]['time'], unit = 'ms')
        dt[i]['internet'] = dt[i]['internet'].astype(float)
        dt[i]['internet']= dt[i]['internet'].fillna(0)   
        dt[i]=dt[i].resample('10T', on='time')['internet'].sum()
        dt[i]=dt[i].reset_index(level=0)
        dt[i].rename(columns={"internet": "%s" %val}, inplace = True)
        dt[i].set_index('time', inplace=True)
        dtt.append(dt[i])

    for i in range(0,len(dtt)):
        t.append(dtt[i])

    dt=pd.concat(t,axis=1)
    dt=dt.fillna(method="bfill")
    dt_reshaped=dt.values.reshape([-1,1])
    scaler = MinMaxScaler(feature_range = (0,1))
    dt_scaled_old=scaler.fit_transform(dt_reshaped)
    dt_scaled_old=dt_scaled_old.reshape(dt.shape[0],dt.shape[1])
    dt_scaled_old=pd.DataFrame(dt_scaled_old)
    nr=int(math.sqrt(dt_scaled_old.shape[1]))
    nc=int(math.sqrt(dt_scaled_old.shape[1]))
    dtt=dt_scaled_old.values.reshape((dt_scaled_old.shape[0], nr, nc))
    
    # Split train data and test data
    train_size = int(len(dtt)*0.8) 
    train_dt, test_dt = dtt[:train_size],dtt[train_size:]
    np.save('../Trained_models/'+'/'+str(damage)+'/Data/test_'+ str(cell)+'.npy', test_dt)

    ds1 = []
    ds2 = []
    ds3 = []
    ds4 = []
    ds = []
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    
    ds1= tf.keras.preprocessing.timeseries_dataset_from_array(train_dt,targets=None, sequence_length = lookback,
                                                              sequence_stride = 1,shuffle = False)
    for batch1 in ds1:
        tmp1.append(batch1.numpy())

    train_X=np.vstack(tmp1)

    ds2= tf.keras.preprocessing.timeseries_dataset_from_array(train_dt[:,2,2],targets=None, sequence_length = 1,
                                                            sequence_stride=1,shuffle = False,start_index = lookback)

    for batch2 in ds2:
        tmp2.append(batch2.numpy())

    train_Y=np.vstack(tmp2)
    train_X=train_X[0:train_Y.shape[0]]

    ds3 = tf.keras.preprocessing.timeseries_dataset_from_array(test_dt,targets = None, sequence_length = lookback,
                                                             sequence_stride = 1,shuffle = False)
    for batch3 in ds3:
        tmp3.append(batch3.numpy())
    test_X = np.vstack(tmp3)

    ds4= tf.keras.preprocessing.timeseries_dataset_from_array(test_dt[:,2,2],targets = None, sequence_length = 1,
                                                                 sequence_stride = 1, shuffle = False, start_index = lookback)
    for batch4 in ds4:
        tmp4.append(batch4.numpy())
    test_Y = np.vstack(tmp4)
    test_X = test_X[0:test_Y.shape[0]]
    train_X = np.reshape(train_X, (train_X.shape[0],train_X.shape[1],train_X.shape[2],train_X.shape[3],1))  
    test_X = np.reshape(test_X, (test_X.shape[0],test_X.shape[1],test_X.shape[2],test_X.shape[3],1))

    nn = make_nn_model(sample_shape, lookback, num_cluster)
    history = nn.fit(train_X, train_Y,batch_size = batch_size, epochs = no_epochs,
                verbose = 0, validation_split = validation_split, shuffle = True)
    
    nn.save('../Trained_models/'+str(damage)+'/Models/mymodel_%d.h5' %cell, save_format='tf')
    dt=[]


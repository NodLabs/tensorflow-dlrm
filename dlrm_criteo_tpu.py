from tensorflow.data import Dataset
from noddlrm.recommenders import DLRM
from tensorflow.keras import optimizers
from tqdm import tqdm
import tensorflow as tf
import dataloader

#setup tpu enviroment
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

raw_data = dataloader.load_criteo('../dataset/')
dim_embed = 4
bottom_mlp_size = [8, 4]
top_mlp_size = [128, 64, 1]
total_iter = int(1e5)
batch_size = 1024
eval_interval = 100
save_interval = eval_interval

# Sample 1000 batches for training
train_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_train'][:batch_size*1000],
                    'sparse_features': raw_data['X_cat_train'][:batch_size*1000],
                    'label': raw_data['y_train'][:batch_size*1000]
                }).batch(batch_size).prefetch(1).shuffle(5*batch_size)
    
# Sample 100 batches for validation
val_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_val'][:batch_size*100],
                    'sparse_features': raw_data['X_cat_val'][:batch_size*100],
                    'label': raw_data['y_val'][:batch_size*100]
             }).batch(batch_size)

optimizer = optimizers.Adam()

dlrm_model = DLRM(
                m_spa=dim_embed,
                ln_emb=raw_data['counts'],
                ln_bot=bottom_mlp_size,
                ln_top=top_mlp_size
             )

auc = tf.keras.metrics.AUC()

@tf.function
def train_step(dense_features, sparse_features, label):
    with tf.GradientTape() as tape:
        loss_value = dlrm_model.get_myloss(dense_features, sparse_features, label)
    gradients = tape.gradient(loss_value, dlrm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dlrm_model.trainable_variables))
    return loss_value

@tf.function
def eval_step(dense_features, sparse_features, label):
    pred = dlrm_model.inference(dense_features, sparse_features)
    auc.update_state(y_true=label, y_pred=pred)

average_loss = tf.keras.metrics.Mean()

for train_iter, batch_data in enumerate(train_dataset):
    
    loss = train_step(**batch_data)
    average_loss.update_state(loss)
    print('%d iter training.' % train_iter, end='\r')
    
    if train_iter % eval_interval == 0:
        for eval_batch_data in tqdm(val_dataset,
                                    leave=False, 
                                    desc='%d iter evaluation' % train_iter):
            eval_step(**eval_batch_data)
        print("Iter: %d, Loss: %.2f, AUC: %.4f" % (train_iter, 
                                                   average_loss.result().numpy(),
                                                   auc.result().numpy()))
        average_loss.reset_states()
        auc.reset_states()

dlrm_model.save('gs://nodtpu/chi/drlm/models/criteo/')

from keras.models import Model

import luna_train_unet5


with tf.device('/gpu:0'):
    model = luna_train_unet5.get_unet()
model.load_weights('./unet5-ex.hdf5-gpu-epoch-50-0p67')

utip_model = Model(input=model.input,
                   output=model.get_layer(name='u_tip').output)

# Now call utip_model.predict(image) with image of [96, 96]

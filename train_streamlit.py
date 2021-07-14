# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 21:30:54 2021

@author: User
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st

def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by <a style='display: block; text-align: center;' >Shubhaditya Goswami</a></p>
    </div>
    """
    return footer


def app():
    """
    Main function that contains the application to train keras based models.
    """
    @tf.function
    def train_step(x, y):
        """
        Tensorflow function to compute gradient, loss and metric defined globally 
        based on given data and model.
        """
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value
    
    
    @tf.function
    def test_step(x, y):
        """
        Tensorflow function to compute predicted loss and metric using sent 
        data from the trained model.
        """
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
        return loss_fn(y, val_logits)
    
    st.title("Keras Training Basic UI")
    st.header("A Streamlit based Web UI To Create And Train Models")
    st.subheader("Create Network:")
    
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    
    in_pl = st.empty()
    input_shape = in_pl.text_input("Enter input shape, either number or tuple")
    if input_shape:
        # Check if input shape is in correct format.
        input_valid = False
        while(not input_valid):
            placeholder = st.empty()
            # Format input_shape.
            if input_shape.isnumeric():
                input_shape = int(input_shape)
                input_valid = True
            elif "(" in input_shape:
                input_shape = eval(input_shape)
                input_valid = True
            else:
                input_shape = in_pl.text_input("Enter input shape, either number or tuple")
                placeholder.write("Invalid input shape.")
        
            
        dense_layer_num = st.number_input("Enter number of dense layers")
    
        if dense_layer_num:
            dense_layer_num = int(dense_layer_num)
            dense_layer_node = st.number_input("Enter number of nodes in dense layers")
    
            if dense_layer_node:
                dense_layer_node = int(dense_layer_node)
                dense_activation = st.text_input("Enter dense layer activation")
                
                if dense_activation:
                    output_num = st.number_input("Enter number of output nodes")
    
                    if output_num:
                        output_num = int(output_num)
                        inputs = keras.Input(shape=(input_shape,), name="digits")
                        dense_layer_dict = {}
                        for i in range(dense_layer_num):
                            if i == 0:
                                dense_layer_dict[i]= layers.Dense(dense_layer_node, 
                                                                  activation=dense_activation)(
                                                                      inputs)
                            else:
                                dense_layer_dict[i] = layers.Dense(dense_layer_node, 
                                                                   activation=dense_activation)(
                                                                       dense_layer_dict[i-1])
                        outputs = layers.Dense(output_num, name="predictions")(dense_layer_dict[i])
                        model = keras.Model(inputs=inputs, outputs=outputs)
                        
                        optim_choice = st.radio("Choose optimizer",("SGD","Adam"))
                        # Instantiate an optimizer.
                        if optim_choice == "SGD":
                            optimizer = keras.optimizers.SGD(learning_rate=1e-3)
                        elif optim_choice == "Adam":
                            optimizer = keras.optimizers.Adam(learning_rate=1e-3)
                        else:
                            optimizer = keras.optimizers.SGD(learning_rate=1e-3)
                            
                        optim_choice = st.radio("Choose loss function",("Categorical crossentropy",
                                                                        "Sparse Categorical crossentropy"))
                        # Instantiate a loss function.
                        if optim_choice == "Categorical crossentropy":
                            loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
                            # Prepare the metrics.
                            train_acc_metric = keras.metrics.CategoricalAccuracy()
                            val_acc_metric = keras.metrics.CategoricalAccuracy()
                        elif optim_choice == "Sparse Categorical crossentropy":
                            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                            # Prepare the metrics.
                            train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
                            val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
                    
                        # Load data.
                        optim_choice = st.radio("Choose dataset",("MNIST", "CIFAR10", "CIFAR100",
                                                                  "IMDB Movie Review", "Reuters Newswire",
                                                                  "Fashion MNIST","Boston Housing"))
                        if optim_choice == "MNIST":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
                        elif optim_choice == "CIFAR10":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
                        elif optim_choice == "CIFAR100":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
                        elif optim_choice == "IMDB Movie Review":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
                        elif optim_choice == "Reuters Newswire":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
                        elif optim_choice == "Fashion MNIST":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
                        elif optim_choice == "Boston Housing":
                            (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
                        
                        # Reshape train and test data.
                        # Prepare the training dataset.
                        batch_size = st.number_input("Enter batch size")
    
                        if batch_size:
                            batch_size = int(batch_size)
                            x_train = np.reshape(x_train, (-1, input_shape))
                            x_test = np.reshape(x_test, (-1, input_shape))
                            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                            
                            # Calculate number of training steps.
                            train_steps_per_epoch = len(x_train) //batch_size
                            
                            # Prepare the validation dataset.
                            # Reserve 10,000 samples for validation.
                            val_ratio = st.number_input("Enter validation ratio")
    
                            if val_ratio:
                                val_ratio = float(val_ratio)
                                val_size = int(val_ratio * x_train.shape[0])
                                x_val = x_train[-val_size:]
                                y_val = y_train[-val_size:]
                                x_train = x_train[:-val_size]
                                y_train = y_train[:-val_size]
                                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                                val_dataset = val_dataset.batch(batch_size)
                                
                                epochs = st.number_input("Enter number of epochs")
    
                                if epochs:
                                    epochs = int(epochs)
                                    save_model = st.text_input("Model name, if want to save model...")
                                    if save_model:
                                        save_condition = st.radio("Choose save condition...",
                                                                  ("train acc","val acc","train loss","val loss"))
                                        
                                    if st.button("Train"):
                                        st.write("Starting training with {} epochs...".format(epochs))
                                        # epochs = 2
                                        for epoch in range(epochs):
                                            print("\nStart of epoch %d" % (epoch,))
                                            st.write("Epoch {}".format(epoch+1))
                                            start_time = time.time()
                                            progress_bar = st.progress(0.0)
                                            percent_complete = 0
                                            epoch_time = 0
                                            # Creating empty placeholder to update each step result in epoch.
                                            st_t = st.empty()
                                            
                                            train_loss_list = []
                                            # Iterate over the batches of the dataset.
                                            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                                                start_step = time.time()
                                                loss_value = train_step(x_batch_train, y_batch_train)
                                                end_step = time.time()
                                                epoch_time += (end_step - start_step)
                                                train_loss_list.append(float(loss_value))
                                                
                                                # Log every 200 batches.
                                                if step % 200 == 0:
                                                    print(
                                                        "Training loss (for one batch) at step %d: %.4f"
                                                        % (step, float(loss_value))
                                                    )
                                                    print("Seen so far: %d samples" % ((step + 1) * batch_size))
                                                    step_acc = float(train_acc_metric.result())
                                                    percent_complete = ((step/train_steps_per_epoch))
                                                    progress_bar.progress(percent_complete)
                                                    st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}"\
                                                     .format((epoch_time),float(step_acc)))
                                            
                                            progress_bar.progress(1.0)
                                        
                                            # Display metrics at the end of each epoch.
                                            train_acc = train_acc_metric.result()
                                            print("Training acc over epoch: %.4f" % (float(train_acc),))
                                        
                                            # Reset training metrics at the end of each epoch
                                            train_acc_metric.reset_states()
                                            
                                            # Find epoch training loss.
                                            print(train_loss_list)
                                            train_loss = round((sum(train_loss_list)/len(train_loss_list)), 5)
                                        
                                            val_loss_list = []
                                            # Run a validation loop at the end of each epoch.
                                            for x_batch_val, y_batch_val in val_dataset: 
                                                val_loss_list.append(float(test_step(x_batch_val, y_batch_val)))
                                            
                                            # Find epoch validation loss.
                                            val_loss = round((sum(val_loss_list)/len(val_loss_list)), 5)
                                        
                                            val_acc = val_acc_metric.result()
                                            val_acc_metric.reset_states()
                                            
                                            print("Validation acc: %.4f" % (float(val_acc),))
                                            print("Time taken: %.2fs" % (time.time() - start_time))
                                            st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}, Validation acc.:{2:.4f}"\
                                                     .format((time.time() - start_time),float(train_acc),float(val_acc)))
    
                                            
                                            # Check if model needs to be saved, and if yess, then with what condition.
                                            if save_model:
                                                if save_condition:
                                                    if epoch == 0:
                                                        best_train_acc = train_acc
                                                        best_train_loss = train_loss
                                                        best_val_loss = val_loss
                                                        best_val_acc = val_acc
                                                        
                                                        # Save first model.
                                                        model.save("./model/"+save_model+".h5", overwrite = True,
                                                                   include_optimizer=True)
                                                        if save_condition in ("train acc","val acc"):
                                                            st.write("Saved model {} as {} increased from 0 to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(train_acc,3) if save_condition == "train acc" else round(val_acc,3)))
                                                        else:
                                                            st.write("Saved model {} as {} decreased from infinite to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(train_loss,3) if save_condition == "train loss" else round(val_loss,3)))
                                                    else:
                                                        if save_condition == "train acc":
                                                            if train_acc >= best_train_acc:
                                                                model.save("./model/"+save_model+".h5", overwrite = True,
                                                                   include_optimizer=True)
                                                                st.write("Saved model {} as {} increased from {} to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(best_train_acc,3),round(train_acc,3)))
                                                                best_train_acc = train_acc
                                                            else:
                                                                st.write("Not saving model as {} did not increase from {}."\
                                                                     .format(save_condition, round(best_train_acc,3)))
                                                        elif save_condition == "val acc":
                                                            if val_acc >= best_val_acc:
                                                                model.save("./model/"+save_model+".h5", overwrite = True,
                                                                   include_optimizer=True)
                                                                st.write("Saved model {} as {} increased from {} to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(best_val_acc,3),round(val_acc,3)))
                                                                best_val_acc = val_acc
                                                            else:
                                                                st.write("Not saving model as {} did not increase from {}."\
                                                                     .format(save_condition, round(best_val_acc,3)))
                                                                    
                                                        elif save_condition == "train loss":
                                                            if train_loss >= best_train_loss:
                                                                model.save("./model/"+save_model+".h5", overwrite = True,
                                                                   include_optimizer=True)
                                                                st.write("Saved model {} as {} decreased from {} to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(best_train_loss,3),round(train_loss,3)))
                                                                best_train_loss = train_loss
                                                            else:
                                                                st.write("Not saving model as {} did not increase from {}."\
                                                                     .format(save_condition, round(best_train_loss,3)))
                                                                    
                                                        elif save_condition == "val loss":
                                                            if val_loss >= best_val_loss:
                                                                model.save("./model/"+save_model+".h5", overwrite = True,
                                                                   include_optimizer=True)
                                                                st.write("Saved model {} as {} decreased from {} to {}."\
                                                                     .format(save_model+".h5", save_condition,
                                                                             round(best_val_loss,3),round(val_loss,3)))
                                                                best_val_loss = val_loss
                                                            else:
                                                                st.write("Not saving model as {} did not increase from {}."\
                                                                     .format(save_condition, round(best_val_loss,3)))
                                                                
                                                    
if __name__=='__main__':
    app()

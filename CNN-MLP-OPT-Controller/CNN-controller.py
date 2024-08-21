from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class SimpleMonitorCNN(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitorCNN, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        body = ev.msg.body

        # Process and save the flow statistics
        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,'
                        'flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,'
                        'packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

            for stat in sorted([flow for flow in body if (flow.priority == 1)], 
                               key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']

                icmp_code, icmp_type, tp_src, tp_dst = -1, -1, 0, 0

                if stat.match['ip_proto'] == 1:
                    icmp_code = stat.match['icmpv4_code']
                    icmp_type = stat.match['icmpv4_type']
                elif stat.match['ip_proto'] == 6:
                    tp_src = stat.match['tcp_src']
                    tp_dst = stat.match['tcp_dst']
                elif stat.match['ip_proto'] == 17:
                    tp_src = stat.match['udp_src']
                    tp_dst = stat.match['udp_dst']

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                try:
                    packet_count_per_second = stat.packet_count / stat.duration_sec
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
                except:
                    packet_count_per_second = 0
                    packet_count_per_nsecond = 0

                try:
                    byte_count_per_second = stat.byte_count / stat.duration_sec
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
                except:
                    byte_count_per_second = 0
                    byte_count_per_nsecond = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},"
                            f"{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},"
                            f"{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{packet_count_per_second},"
                            f"{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond}\n")

    def flow_training(self):
        self.logger.info("CNN Training ...")
        modelDB = 'cnn_model_detech.h5'

        if os.path.exists(modelDB):
            print("File exists, loading model")
            self.flow_model = Sequential.load_model(modelDB)
        else:
            print("File does not exist, training model")

            flow_dataset = pd.read_csv('FlowStatsfile.csv')

            # Data preprocessing
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

            X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
            y_flow = flow_dataset.iloc[:, -1].values

            # Normalize the data
            X_flow = (X_flow - np.mean(X_flow, axis=0)) / np.std(X_flow, axis=0)

            # Reshape for CNN
            X_flow = X_flow.reshape(X_flow.shape[0], X_flow.shape[1], 1, 1)

            # Convert labels to categorical
            y_flow = to_categorical(y_flow)

            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

            # CNN model
            self.flow_model = Sequential()

            # Layer 1
            self.flow_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X_flow.shape[1], 1, 1)))
            self.flow_model.add(MaxPooling2D(pool_size=(2, 2)))

            # Layer 2
            self.flow_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            self.flow_model.add(MaxPooling2D(pool_size=(2, 2)))

            # Layer 3
            self.flow_model.add(Flatten())
            self.flow_model.add(Dense(128, activation='relu'))

            # Output layer
            self.flow_model.add(Dense(y_flow.shape[1], activation='softmax'))

            # Compile the model
            self.flow_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            self.flow_model.fit(X_flow_train, y_flow_train, epochs=10, batch_size=32, validation_data=(X_flow_test, y_flow_test))

            # Save the model
            self.flow_model.save(modelDB)

            # Evaluate the model
            y_flow_pred = self.flow_model.predict(X_flow_test)
            y_flow_pred_train = self.flow_model.predict(X_flow_train)

            y_flow_pred = np.argmax(y_flow_pred, axis=1)
            y_flow_test = np.argmax(y_flow_test, axis=1)
            y_flow_pred_train = np.argmax(y_flow_pred_train, axis=1)
            y_flow_train = np.argmax(y_flow_train, axis=1)

            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("Confusion Matrix")
            cm = confusion_matrix(y_flow_test, y_flow_pred)
            self.logger.info(cm)

            acc = accuracy_score(y_flow_test, y_flow_pred)
            acc_train = accuracy_score(y_flow_train, y_flow_pred_train)
            print("Training Accuracy: ", acc_train)

            self.logger.info("Test Accuracy = {0:.2f} %".format(acc * 100))
            fail = 1.0 - acc
            self.logger.info("Test Fail Rate = {0:.2f} %".format(fail * 100))
            self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict_flow = predict_flow_dataset.iloc[:, :].values.astype('float64')

            # Normalize the data
            X_predict_flow = (X_predict_flow - np.mean(X_predict_flow, axis=0)) / np.std(X_predict_flow, axis=0)

            # Reshape for CNN
            X_predict_flow = X_predict_flow.reshape(X_predict_flow.shape[0], X_predict_flow .shape[1], 1, 1)

	            # Predict
        y_predict_flow = self.flow_model.predict(X_predict_flow)

        print(y_predict_flow)

    except Exception as e:
        self.logger.info("No records to predict!")
        print("Error: ", e)


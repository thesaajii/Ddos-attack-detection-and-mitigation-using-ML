from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.base import app_manager

import switch
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix, accuracy_score
import shap


class CombinedCNNMLPDetector(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CombinedCNNMLPDetector, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.model = None
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
        file0 = open("PredictFlowStatsfile.csv", "w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            tp_src, tp_dst, icmp_code, icmp_type = 0, 0, -1, -1

            if ip_proto == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
            elif ip_proto == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']
            elif ip_proto == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

            packet_count_per_second = stat.packet_count / stat.duration_sec if stat.duration_sec > 0 else 0
            packet_count_per_nsecond = stat.packet_count / stat.duration_nsec if stat.duration_nsec > 0 else 0
            byte_count_per_second = stat.byte_count / stat.duration_sec if stat.duration_sec > 0 else 0
            byte_count_per_nsecond = stat.byte_count / stat.duration_nsec if stat.duration_nsec > 0 else 0

            file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{packet_count_per_second},{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond}\n")
        
        file0.close()

    def flow_training(self):
        self.logger.info("Training Combined CNN-MLP Model...")
        modelDB = 'cnn_mlp_model.pkl'
        
        if os.path.exists(modelDB):
            print("Model file exists, loading model...")
            with open(modelDB, 'rb') as file:
                self.model = pickle.load(file)
        else:
            print("Model file does not exist, training model...")
            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

            X = flow_dataset.iloc[:, :-1].values.astype('float64')
            y = flow_dataset.iloc[:, -1].values

            # Data normalization
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

            # Reshape data for CNN
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

            # Define CNN model
            cnn_model = Sequential([
                Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], 1, 1), padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')  # Binary classification
            ])
            cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train CNN model
            cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=64, verbose=1)

            # Combine with MLP
            self.model = Sequential([
                cnn_model,
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train combined model
            self.model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

            # Save model
            with open(modelDB, 'wb') as file:
                pickle.dump(self.model, file)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            print("Confusion Matrix:\n", cm)
            print("Accuracy:", acc * 100, "%")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict = predict_flow_dataset.values.astype('float64')
            X_predict = StandardScaler().fit_transform(X_predict)

            # Reshape for CNN
            X_predict_cnn = X_predict.reshape(X_predict.shape[0], X_predict.shape[1], 1, 1)

            y_pred = self.model.predict(X_predict_cnn)
            y_pred = (y_pred > 0.5).astype(int)

            legitimate_traffic = np.sum(y_pred == 0)
            ddos_traffic = np.sum(y_pred == 1)

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_traffic / len(y_pred)) * 100 > 80:
                self.logger.info("Legitimate traffic detected...")
            else:
                victim = int(predict_flow_dataset.iloc[ddos_traffic, 5]) % 20
                self.logger.info("DDoS traffic detected...")
                self.logger.info(f"Victim is host: h{victim}")
            self.logger.info("------------------------------------------------------------------------------")

            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write('timestamp,datapath_id,flow_id ,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
	        except Exception as e:
        print("Error in prediction: ", str(e))	
	def main():
	app_manager.run_apps([CombinedCNNMLPDetector])

	if name == "main":
	main()


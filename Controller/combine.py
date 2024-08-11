from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import switch
from datetime import datetime
import pickle, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.new_flow_stats = False  # Flag to indicate new flow stats
        self.feature_columns = None  # Store feature columns for consistency
        self.voting_classifier = None

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

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
            if self.new_flow_stats:
                self.flow_predict()
                self.new_flow_stats = False  # Reset the flag after prediction

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        self.new_flow_stats = True  # Set the flag indicating new flow stats

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file_path = "PredictFlowStatsfile.csv"
        with open(file_path, "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,'
                        'flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,'
                        'byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,'
                        'byte_count_per_nsecond\n')
            body = ev.msg.body
            icmp_code = -1
            icmp_type = -1
            tp_src = 0
            tp_dst = 0

            for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow:
                    (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']

                if stat.match['ip_proto'] == 1:
                    icmp_code = stat.match['icmpv4_code']
                    icmp_type = stat.match['icmpv4_type']
                elif stat.match['ip_proto'] == 6:
                    tp_src = stat.match['tcp_src']
                    tp_dst = stat.match['tcp_dst']
                elif stat.match['ip_proto'] == 17:
                    tp_src = stat.match['udp_src']
                    tp_dst = stat.match['udp_dst']

                flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)

                try:
                    packet_count_per_second = stat.packet_count / stat.duration_sec
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
                except ZeroDivisionError:
                    packet_count_per_second = 0
                    packet_count_per_nsecond = 0

                try:
                    byte_count_per_second = stat.byte_count / stat.duration_sec
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
                except ZeroDivisionError:
                    byte_count_per_second = 0
                    byte_count_per_nsecond = 0

                file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst, stat.match['ip_proto'],
                    icmp_code, icmp_type, stat.duration_sec, stat.duration_nsec, stat.idle_timeout, stat.hard_timeout,
                    stat.flags, stat.packet_count, stat.byte_count, packet_count_per_second, packet_count_per_nsecond,
                    byte_count_per_second, byte_count_per_nsecond))

    def flow_training(self):
        self.logger.info("Flow Training ...")
        dt_modelDB = 'dt_model_detech.pkl'
        knn_modelDB = 'knn_model_detech.pkl'

        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '', regex=False)
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '', regex=False)
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '', regex=False)

        # Ensure that the features are numeric
        for column in flow_dataset.columns[:-1]:
            flow_dataset[column] = pd.to_numeric(flow_dataset[column], errors='coerce')

        self.feature_columns = flow_dataset.columns[:-1]  # Store feature columns for consistency

        X_flow = flow_dataset[self.feature_columns].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        if os.path.exists(dt_modelDB) and os.path.exists(knn_modelDB):
            print("Files exist, loading models")
            with open(dt_modelDB, 'rb') as dt_file, open(knn_modelDB, 'rb') as knn_file:
                dt_model = pickle.load(dt_file)
                knn_model = pickle.load(knn_file)
        else:
            print("Files do not exist, training models")
            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

            # Train Decision Tree
            dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt_model = dt_classifier.fit(X_flow_train, y_flow_train)
            y_flow_pred_dt = dt_model.predict(X_flow_test)

            with open(dt_modelDB, 'wb') as dt_file:
                pickle.dump(dt_model, dt_file)

            self.logger.info("Decision Tree Confusion Matrix")
            cm_dt = confusion_matrix(y_flow_test, y_flow_pred_dt)
            self.logger.info(cm_dt)

            acc_dt = accuracy_score(y_flow_test, y_flow_pred_dt)
            self.logger.info("Decision Tree Accuracy = {0:.2f} %".format(acc_dt * 100))

            # Train KNN
            knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            knn_model = knn_classifier.fit(X_flow_train, y_flow_train)
            y_flow_pred_knn = knn_model.predict(X_flow_test)

            with open(knn_modelDB, 'wb') as knn_file:
                pickle.dump(knn_model, knn_file)

            self.logger.info("KNN Confusion Matrix")
            cm_knn = confusion_matrix(y_flow_test, y_flow_pred_knn)
            self.logger.info(cm_knn)

            acc_knn = accuracy_score(y_flow_test, y_flow_pred_knn)
            self.logger.info("KNN Accuracy = {0:.2f} %".format(acc_knn * 100))

        # Combine both models in a Voting Classifier
        self.voting_classifier = VotingClassifier(estimators=[
            ('dt', dt_model),
            ('knn', knn_model)
        ], voting='hard')

        # Train the voting classifier on the full dataset (for simplicity)
        self.voting_classifier.fit(X_flow, y_flow)

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '', regex=False)
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '', regex=False)
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '', regex=False)

            # Ensure that the features are numeric
            for column in predict_flow_dataset.columns:
                predict_flow_dataset[column] = pd.to_numeric(predict_flow_dataset[column], errors='coerce')

            # Align features for prediction
            predict_flow_dataset = predict_flow_dataset[self.feature_columns]

            # Predict using the Voting Classifier
            if predict_flow_dataset.shape[0] > 0:  # Ensure there is data to predict
                X_predict_flow = predict_flow_dataset.values
                X_predict_flow = X_predict_flow.astype('float64')

                y_flow_pred = self.voting_classifier.predict(X_predict_flow)

                # Calculate the percentage of legitimate traffic
                legitimate_count = (y_flow_pred == 0).sum()  # Assuming 0 indicates legitimate traffic
                total_count = len(y_flow_pred)
                legitimate_percentage = (legitimate_count / total_count) * 100

                # Log whether the traffic is legitimate or DDoS
                if legitimate_percentage > 80:
                    self.logger.info("Legitimate traffic ...")
                else:
                    self.logger.info("DDoS traffic ...")
                    
                    # Identify and log victim IPs
                    for index, row in predict_flow_dataset.iterrows():
                        if y_flow_pred[index] == 1:  # Assuming 1 indicates DDoS
                            victim_ip = row['ip_dst']
                            self.logger.info("Victim is host: h{}".format(victim_ip))
                            self.mitigate_ddos(ev.msg.datapath, victim_ip)  # Call mitigation function
            
        except Exception as e:
            # Commented out to disable error output
            # self.logger.error("Prediction error: {}".format(e))
            pass

    def mitigate_ddos(self, datapath, victim_ip):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # Construct a match object to match packets with the victim IP address as destination
        match = parser.OFPMatch(ipv4_dst=victim_ip, eth_type=0x0800)  # IPv4 traffic

        # Create an action to drop the matched packets
        actions = []

        # Create a flow mod message to add the drop rule
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_CLEAR_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=10,
                                match=match, instructions=inst)
        datapath.send_msg(mod)
        self.logger.info("Mitigation: Dropping traffic to victim IP {}".format(victim_ip))



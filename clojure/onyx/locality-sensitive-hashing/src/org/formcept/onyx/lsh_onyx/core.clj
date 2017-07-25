;;
;; Copyright (c) 2011-onwards, FORMCEPT [http://www.formcept.com]
;; All rights reserved.
;;
;; NOTICE:  All information contained herein is, and remains
;; the property of FORMCEPT and its suppliers, if any.
;; The intellectual and technical concepts contained
;; herein are proprietary to FORMCEPT and its suppliers and
;; may be covered by U.S. and Foreign Patents, patents in process,
;; and are protected by trade secret or copyright law. Dissemination
;; of this information or reproduction of this material
;; is strictly forbidden unless prior written permission is obtained
;; from FORMCEPT.
;;

(ns org.formcept.onyx.lsh-onyx.core
  (:require [clojure.core.async :refer [chan >!! <!! close!]]
            [onyx.extensions :as extensions]
            [onyx.plugin.core-async :refer [take-segments!]]
            [onyx.api]
            [org.formcept.onyx.onyx-utils.aggregation.core]
            [org.formcept.onyx.onyx-utils.string-utils.core :refer [band-hash levenstein-distance]])
  (:gen-class))

(defn add-hash-to-records
  "Adds a new k-v pair (hash,<hash>) to the original data"
  [{:keys [index data] :as segment}]
  (mapv #(conj segment {:hash %}) (band-hash data 2 5 2)))

(defn gen-cartesian-product
  "Generates row pairs from the original data"
  [data]
  (->> (flatten 
         (map-indexed 
           (fn [idxi i] 
             (map-indexed 
               (fn [idxj j] 
                 (if (> idxj idxi) (assoc {} :original i :candidate j))) data)) data))
       (filter #(not (nil? %)))))

(defn cartesian-products
  "Gets all the row pairs from the input segments"
  [segment]
  (->> (vals segment)
       (reduce #(into %1 %2))
       (mapv #(dissoc % :hash))
       (into #{})
       (sort-by :index)
       gen-cartesian-product
       (filterv #(not (empty? %)))))

(defn find-levenstein-distance
  "Finds normalised levenshtein distance between two strings `a` and `b`"
  [a b]
  (double (/ (levenstein-distance a b) (+ (count a) (count b)))))

(defn compare-segment-pairs
  "Calculates similarity between pairs of segments"
  [{:keys [distinct] :as segment}]
  (assoc {} :original_index (get-in distinct [:original :index])
            :original_data (get-in distinct [:original :data])
            :duplicate_index (get-in distinct [:candidate :index])
            :duplicate_data (get-in distinct [:candidate :data])
            :error (find-levenstein-distance
                     (get-in distinct [:original :data])
                     (get-in distinct [:candidate :data]))))

(def workflow
  [[:in :gen-hash]
   [:gen-hash :combine-by-hash]
   [:combine-by-hash :gen-candidates]
   [:gen-candidates :distinct-candidates]
   [:distinct-candidates :calculate-similarity]
   [:calculate-similarity :out]])

(def capacity 1000)

(def input-chan (chan capacity))
(def input-buffer (atom {}))

(def output-chan (chan capacity))

(def batch-size 1000)

(defn catalog
  [params]
  [{:onyx/name :in
    :onyx/plugin :onyx.plugin.core-async/input
    :onyx/type :input
    :onyx/medium :core.async
    :onyx/batch-size batch-size
    :onyx/max-peers 1
    :onyx/doc "Reads segments from a core.async channel"}
    
   {:onyx/name :gen-hash
    :onyx/fn ::add-hash-to-records
    :onyx/type :function
    :onyx/batch-size batch-size
    :onyx/doc "Generate hashes for every record"}
   
   {:onyx/name :combine-by-hash
    :onyx/fn :clojure.core/identity
    :onyx/type :function
    :onyx/batch-size batch-size
    :onyx/doc "Combine by hashes"}
    
   {:onyx/name :gen-candidates
    :onyx/fn ::cartesian-products
    :onyx/type :function
    :onyx/batch-size batch-size
    :onyx/doc "Generate candidate pairs"}
   
   {:onyx/name :distinct-candidates
    :onyx/fn :clojure.core/identity
    :onyx/type :function
    :onyx/batch-size batch-size
    :onyx/doc "Applying overall distinct i.e. on the entire dataset"}
   
   {:onyx/name :calculate-similarity
    :onyx/fn ::compare-segment-pairs
    :onyx/type :function
    :onyx/batch-size batch-size
    :onyx/doc "Calculate Similarity between candidates"}
   
   {:onyx/name :out
    :onyx/plugin :onyx.plugin.core-async/output
    :onyx/type :output
    :onyx/medium :core.async
    :onyx/max-peers 1
    :onyx/batch-size batch-size
    :onyx/doc "Writes segments to a core.async channel"}])

(defn windows
  [params]
  [{:window/id :combine-by-hash-window
    :window/task :combine-by-hash
    :window/type :global
    :window/aggregation [:onyx.windowing.aggregation/collect-by-key :hash]}
   {:window/id :distinct-candidates-window
    :window/task :distinct-candidates
    :window/type :global
    :window/aggregation :org.formcept.onyx.onyx-utils.aggregation.core/onyx-distinct}])

(defn triggers
  [params]
  [{:trigger/window-id :combine-by-hash-window
    :trigger/id :combine-by-hash-trigger
    :trigger/refinement :onyx.refinements/accumulating
    :trigger/on :onyx.triggers/segment
    :trigger/threshold [2 :elements]
    :trigger/emit ::dump-window!}
   {:trigger/window-id :distinct-candidates-window
    :trigger/id :distinct-candidates-trigger
    :trigger/refinement :onyx.refinements/accumulating
    :trigger/on :onyx.triggers/segment
    :trigger/threshold [2 :elements]
    :trigger/emit ::dump-window!}])

(defn remove-unhashed-records
  [event old-segment new-segment all-new]
  (contains? new-segment :hash))

(defn remove-uncollected-records
  [event old-segment new-segment all-new]
  (not (contains? new-segment :data)))

(defn keep-only-distinct-candidates
  [event old-segment new-segment all-new]
  (contains? new-segment :distinct))

(defn crosses-threshold?
  [event old-segment new-segment all-new ld-threshold]
  (<= (new-segment :error) ld-threshold))

(defn flow-conditions
  [params]
  [{:flow/from :gen-hash
    :flow/to [:combine-by-hash]
    :flow/predicate ::remove-unhashed-records}
   {:flow/from :combine-by-hash
    :flow/to [:gen-candidates]
    :flow/predicate ::remove-uncollected-records}
   {:flow/from :distinct-candidates
    :flow/to [:calculate-similarity]
    :flow/predicate ::keep-only-distinct-candidates}
   {:flow/from :calculate-similarity
    :flow/to [:out]
    :lsh-onyx/ld-threshold (:ld-threshold params)
    :flow/predicate [::crosses-threshold? :lsh-onyx/ld-threshold]}])

(defn dump-window!
  [event window trigger {:keys [lower-bound upper-bound] :as window-data} state]
  state)

(def input-segments
  [{:index 0 :data "punit naik"} {:index 1 :data "punit"}])

(defn publish-to-chan
  "Write data to input channel and close it"
  []
  (do
    (doseq [segment input-segments]
      (>!! input-chan segment))
    (close! input-chan)))

(def id (java.util.UUID/randomUUID))

(def always-true (constantly true))

(def env-config
  {:zookeeper/address "127.0.0.1:2188"
   :zookeeper/server? true
   :zookeeper.server/port 2188
   :onyx/tenancy-id id})

(def peer-config
  {:zookeeper/address "127.0.0.1:2188"
   :onyx/tenancy-id id
   :onyx.peer/job-scheduler :onyx.job-scheduler/balanced
   :onyx.messaging/impl :aeron
   :onyx.messaging/peer-port 40200
   :onyx.messaging/bind-addr "localhost"})

(defn inject-in-ch [event lifecycle]
  {:core.async/buffer input-buffer
   :core.async/chan input-chan})

(defn inject-out-ch [event lifecycle]
  {:core.async/chan output-chan})

(def in-calls
  {:lifecycle/before-task-start inject-in-ch})

(def out-calls
  {:lifecycle/before-task-start inject-out-ch})

(def lifecycles
  [{:lifecycle/task :in
    :lifecycle/calls ::in-calls}
   {:lifecycle/task :in
    :lifecycle/calls :onyx.plugin.core-async/reader-calls}
   {:lifecycle/task :out
    :lifecycle/calls ::out-calls}
   {:lifecycle/task :out
    :lifecycle/calls :onyx.plugin.core-async/writer-calls}])

(defn run
  "Performs startup, runs the code, shuts down the env and fetches the result as well"
  [params]
  (let [env (onyx.api/start-env env-config)
        peer-group (onyx.api/start-peer-group peer-config)
        n-peers (count (set (mapcat identity workflow)))
        v-peers (onyx.api/start-peers n-peers peer-group)]
    (do
      (publish-to-chan)
      (as-> (onyx.api/submit-job peer-config
                                 {:workflow workflow
                                  :catalog (catalog params)
                                  :lifecycles lifecycles
                                  :windows (windows params)
                                  :triggers (triggers params)
                                  :flow-conditions (flow-conditions params)
                                  :task-scheduler :onyx.task-scheduler/balanced}) $
            (onyx.api/await-job-completion peer-config (:job-id $)))
      (doseq [v-peer v-peers]
        (onyx.api/shutdown-peer v-peer))
      (onyx.api/shutdown-peer-group peer-group)
      (onyx.api/shutdown-env env)
      (onyx.plugin.core-async/take-segments! output-chan 50))))

(defn -main
  [& args]
  (println (run {:ld-threshold (or (Float/parseFloat (first args)) 0.10)})))


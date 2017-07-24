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

(ns org.formcept.onyx.onyx-utils.aggregation.core
  (:gen-class))

(defn distinctv
  [v]
  (vec (distinct v)))

(defn distinct-aggregation-fn-init [window]
  [])

(defn distinct-aggregation-apply-log [window state v]
  (distinctv (conj state (assoc {} :distinct v))))

(defn distinct-on-key-aggregation-apply-log [window state v]
  (let [k (second (:window/aggregation window))]
    (distinctv (conj state (assoc {} :distinct (get v k))))))

(defn distinct-aggregation-fn [window state segment]
  segment)

(defn distinct-super-aggregation [window state-1 state-2]
  (distinctv (into (distinctv state-1) (distinctv state-2))))

(def ^:export onyx-distinct
  {:aggregation/init distinct-aggregation-fn-init
   :aggregation/create-state-update distinct-aggregation-fn
   :aggregation/apply-state-update distinct-aggregation-apply-log
   :aggregation/super-aggregation-fn distinct-super-aggregation})

(def ^:export onyx-distinct-on-key
  {:aggregation/init distinct-aggregation-fn-init
   :aggregation/create-state-update distinct-aggregation-fn
   :aggregation/apply-state-update distinct-on-key-aggregation-apply-log
   :aggregation/super-aggregation-fn distinct-super-aggregation})


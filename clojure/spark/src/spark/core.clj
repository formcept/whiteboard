(ns spark.core
  (:require [clojure.string :as s]
            [flambo [api :as f] [conf :as conf] [session :as fs] [tuple :as ft]])
  (:import [org.apache.spark.api.java JavaSparkContext])
  (:gen-class))

(defn- get-deps-jars
  []
  (->> (ClassLoader/getSystemClassLoader)
       (.getURLs)
       (map #(.getPath %))))

(defn- get-spark-session
  "Gets or creates a Spark session"
  []
  (-> (fs/session-builder)
      (fs/config (conf/spark-conf))
      (fs/master "spark://172.17.0.1:7077")
      (fs/app-name "Spark Word Count")
      (fs/config "spark.jars" "target/spark-0.1.0-SNAPSHOT-standalone.jar")
      ;; include all JARs of system classloader
      ;;(fs/config (-> (conf/spark-conf)
      ;;               (conf/jars (get-deps-jars))))
      ;; include specific JARs by class
      ;;(fs/config "spark.jars" (-> flambo.kryo.BaseFlamboRegistrator
      ;;                            .getProtectionDomain
      ;;                            .getCodeSource
      ;;                            .getLocation
      ;;                            .getPath))
      (fs/get-or-create)))

(defn run-spark-job
  []
  (-> (get-spark-session)
      (.sparkContext)
      (JavaSparkContext.)
      (f/parallelize ["apache spark is a fast and general purpose cluster computing system"
                      "flambo is a clojure dsl for spark"
                      "create and manipulate spark data structures using idiomatic clojure"
                      "flambo makes developing spark applications quick and painless"
                      "flambo utilizes the powerful abstractions available in clojure"])
      (f/flat-map (f/iterator-fn [r] (s/split r #"\s")))
      (f/map-to-pair (f/fn [w] (ft/tuple w 1)))
      (f/reduce-by-key (f/fn [wc1 wc2] (+ wc1 wc2)))
      (f/collect)))

(defn -main
  [& args]
  (->> (run-spark-job)
       (clojure.tools.logging/info)))
